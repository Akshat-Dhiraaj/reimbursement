"""Tests for the simple LLM-judge validity pipeline (llm_validate.py) — offline.

The API calls hit the network, so we don't make them here. We pin the contract: the external
prompt loads and covers the checks, provider resolution follows the keys, a PDF routes natively
to Gemini vs is rasterised for image-only Groq, and validate() parses the model's JSON (failing
safe to "review"). The live call is exercised by hand: `slipguard validate <receipt> --provider groq`.
"""

from __future__ import annotations

import pytest

from slipguard import llm_validate as L


def test_prompt_loads_and_covers_the_checks():
    low = L.load_prompt().lower()
    for needle in ("ai", "edit", "date", "arithmetic", "vendor", "decision", "json", "review"):
        assert needle in low


def test_is_pdf():
    assert L._is_pdf("a.PDF") and not L._is_pdf("a.jpg")


def test_import_order_has_no_circular_import():
    """Regression: uvicorn loads ``web.api`` -> ``llm_validate`` FIRST, so importing this module
    before the extractors package must not deadlock on a circular import (``llm_validate`` reaches
    into ``extractors.vlm_qwen`` while the registry's ``groq_vlm`` reaches back into ``llm_validate``).
    pytest imports the extractors package early, which MASKS the cycle — so check the real
    uvicorn order in a fresh subprocess."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", "import slipguard.llm_validate, slipguard.extractors"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_resolve_provider_follows_keys(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "GROQ_API_KEY_2", "LMSTUDIO_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        L.resolve_provider("auto")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert L.resolve_provider("auto") == "groq"
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert L.resolve_provider("auto") == "groq"             # groq wins (more quota + multi-key) when both set
    assert L._provider_chain("auto") == ["groq", "gemini"]  # the full fallback order
    assert L.resolve_provider("gemini") == "gemini"         # explicit choice overrides auto


def test_pdf_goes_native_to_gemini(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7 minimal")
    parts = L._parts_for(str(pdf), "gemini")           # no pypdfium2 needed on the Gemini path
    assert len(parts) == 1 and parts[0][1] == "application/pdf"


def _jpg(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    p = tmp_path / "r.jpg"
    Image.new("RGB", (16, 16), "white").save(p, "JPEG")
    return str(p)


def test_validate_parses_verdict(monkeypatch, tmp_path):
    img = _jpg(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(L, "_call_groq", lambda prompt, parts, model:
                        '{"decision":"review","date":"2026-01-10",'
                        '"ai_or_edit_suspected":false,"summary":"ok"}')
    v = L.validate(img, provider="groq")
    assert v["decision"] == "review" and v["date"] == "2026-01-10"
    assert v["_provider"] == "groq" and v["_path"] == img


def test_validate_fails_safe_to_review_on_garbage(monkeypatch, tmp_path):
    img = _jpg(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(L, "_call_groq", lambda *a: "sorry, I can't return JSON")
    v = L.validate(img, provider="groq")
    assert v["decision"] == "review" and "_raw" in v   # unparseable -> human looks


def test_post_json_caps_backoff_and_gives_up(monkeypatch):
    # A *daily* limit can return a huge retry-after; the cap must keep each sleep <= _MAX_BACKOFF and
    # then give up (raise) rather than blocking for hours — the batch-eval "hang" this guards against.
    import urllib.error
    import urllib.request
    slept: list[float] = []
    monkeypatch.setattr(L.time, "sleep", lambda s: slept.append(s))

    def always_429(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 429, "rate limit", {"retry-after": "3600"}, None)

    monkeypatch.setattr(urllib.request, "urlopen", always_429)
    with pytest.raises(urllib.error.HTTPError):
        L._post_json("https://x", {}, {}, retries=2)
    assert slept and all(s <= L._MAX_BACKOFF for s in slept)   # 3600s retry-after → capped, not slept


# --- the local LM Studio provider --------------------------------------------

def test_lmstudio_requires_a_model(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)
    with pytest.raises(RuntimeError):           # no --model and no LMSTUDIO_MODEL → clear error
        L._call_lmstudio("prompt", [], None)


def test_validate_dispatches_to_lmstudio(monkeypatch, tmp_path):
    img = _jpg(tmp_path)
    monkeypatch.setattr(L, "_call_lmstudio",
                        lambda prompt, parts, model: '{"decision":"approve","total":5.0,"summary":"ok"}')
    v = L.validate(img, provider="lmstudio", model="x/y", cross_check=False)
    assert v["decision"] == "approve" and v["_provider"] == "lmstudio"   # routed to the local caller


# --- Groq multi-key fallback (rotate when one hits its rate/daily limit) ------

def test_numbered_keys_fallback_order(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    assert L._numbered_keys("GROQ_API_KEY") == ["k1", "k2"]
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    assert L._numbered_keys("GROQ_API_KEY") == []


def test_call_groq_rotates_to_second_key_on_429(monkeypatch):
    import urllib.error
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    seen = []

    def fake_post(url, body, headers, timeout=90.0, retries=4):
        key = headers["Authorization"].split()[-1]
        seen.append((key, retries))
        if key == "k1":
            raise urllib.error.HTTPError(url, 429, "rate limit", {}, None)   # first key is out
        return {"choices": [{"message": {"content": '{"decision":"approve"}'}}]}

    monkeypatch.setattr(L, "_post_json", fake_post)
    out = L._call_groq("p", [], None)
    assert out == '{"decision":"approve"}'
    assert [k for k, _ in seen] == ["k1", "k2"]          # rotated to the 2nd key
    assert seen[0][1] == 0 and seen[1][1] == 4           # non-last fast (retries=0); last backs off


def test_validate_falls_back_across_providers_on_429(monkeypatch, tmp_path):
    # auto chain is groq -> gemini; if groq is rate-limited, validate() falls back to gemini.
    import urllib.error
    img = _jpg(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)

    def groq_429(*a):
        raise urllib.error.HTTPError("u", 429, "rate limit", {}, None)

    monkeypatch.setattr(L, "_call_groq", groq_429)
    monkeypatch.setattr(L, "_call_gemini", lambda *a: '{"decision":"approve"}')
    v = L.validate(img, cross_check=False)               # auto: groq(429) -> gemini(ok)
    assert v["decision"] == "approve" and v["_provider"] == "gemini"


# --- the deterministic cross-check refinement (#85) --------------------------

def test_reconcile_escalates_on_broken_arithmetic():
    # LLM said 'approve', but its OWN numbers don't reconcile -> overruled (never relaxed)
    out = L.reconcile({"decision": "approve", "subtotal": 100.0, "tax": 10.0, "total": 999.0}, "x.json")
    assert out["decision"] in ("review", "reject")
    assert out["llm_decision"] == "approve" and out["deterministic_decision"] in ("review", "reject")


def test_reconcile_escalates_on_future_date():
    out = L.reconcile({"decision": "approve", "date": "2099-01-01"}, "x.json")
    assert out["decision"] != "approve"


def test_reconcile_keeps_clean_approve():
    # arithmetic reconciles and nothing else fires -> the deterministic layer agrees
    out = L.reconcile({"decision": "approve", "subtotal": 100.0, "tax": 18.0, "total": 118.0}, "x.json")
    assert out["decision"] == "approve" and out["deterministic_decision"] == "approve"


def test_reconcile_never_downgrades_llm_reject():
    # LLM 'reject' (it saw something visual) + clean math -> stays reject (we take the stricter)
    out = L.reconcile({"decision": "reject", "subtotal": 100.0, "tax": 18.0, "total": 118.0}, "x.json")
    assert out["decision"] == "reject"


def test_reconcile_adds_score_breakdown():
    # the per-detector score x weight breakdown the web UI renders
    bd = L.reconcile({"decision": "approve", "subtotal": 100.0, "tax": 10.0, "total": 999.0},
                     "x.json")["_breakdown"]
    assert {"risk_score", "review_at", "reject_at", "signals"} <= bd.keys()
    names = {s["detector"] for s in bd["signals"]}
    assert {"arithmetic", "date_sanity", "duplicate"}.issubset(names)
    arith = next(s for s in bd["signals"] if s["detector"] == "arithmetic")
    assert arith["abstained"] is False and arith["score"] > 0.5      # broken total -> high fraud score
    assert 0.0 <= bd["risk_score"] <= 1.0


def test_verdict_to_receipt_maps_fields():
    r = L._verdict_to_receipt({"vendor": "Croma", "date": "2026-01-10", "subtotal": 100.0,
                               "tax": 18.0, "total": 118.0, "service_charge": 5.0,
                               "tax_id": "27AAPFU0939F1ZV", "country": "IN"}, "x.json")
    assert r.vendor_name == "Croma" and r.total == 118.0 and r.service_charge == 5.0
    assert r.vendor_tax_id == "27AAPFU0939F1ZV" and r.country == "IN"
