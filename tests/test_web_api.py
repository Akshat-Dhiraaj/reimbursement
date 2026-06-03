"""Tests for the web UI backend (the [web] extra).

The whole module skips if FastAPI is not installed (web is optional). The response-shaping logic
(``_shape`` / ``_num``) is pure and tested directly; the HTTP endpoints are exercised with Starlette's
TestClient and the network-bound ``validate()`` monkeypatched, so these run fully offline.
"""

import pytest

pytest.importorskip("fastapi")

from slipguard.web import api  # noqa: E402


def _approve_verdict() -> dict:
    return {
        "decision": "approve", "confidence": 0.9, "summary": "Looks fine.",
        "deterministic_reasons": ["[arithmetic] all arithmetic reconciles"],
        "red_flags": [],
        "vendor": "Acme", "date": "2024-01-02", "currency": "USD",
        "subtotal": 10.0, "tax": 0.8, "total": 10.8,
        "ai_or_edit_suspected": False, "date_valid": True, "arithmetic_consistent": True,
        "ai_or_edit_signs": [], "llm_decision": "approve", "deterministic_decision": "approve",
        "_provider": "groq",
    }


# --- pure shaping ---------------------------------------------------------------------------

def test_num_coerces():
    assert api._num(1) == 1.0
    assert api._num("0.5") == 0.5
    assert api._num(None) is None
    assert api._num(True) is None        # bools are not confidences
    assert api._num("not-a-number") is None


def test_shape_approve_maps_fields_and_reasons():
    s = api._shape(_approve_verdict(), "r.png")
    assert s["approved"] is True
    assert s["decision"] == "approve"
    assert s["confidence"] == 0.9
    assert {"source": "deterministic",
            "text": "[arithmetic] all arithmetic reconciles"} in s["reasons"]
    assert s["fields"]["vendor"] == "Acme"
    assert s["fields"]["total"] == 10.8
    assert s["checks"]["arithmetic_consistent"] is True
    assert s["filename"] == "r.png" and s["provider"] == "groq"


def test_shape_not_approved_tags_both_sources():
    v = _approve_verdict()
    v.update(decision="reject", llm_decision="reject", deterministic_decision="review",
             red_flags=["future date"],
             deterministic_reasons=["[date_sanity] date is in the future"])
    s = api._shape(v, "bad.jpg")
    assert s["approved"] is False
    assert {r["source"] for r in s["reasons"]} == {"model", "deterministic"}
    assert "future date" in [r["text"] for r in s["reasons"]]


def test_shape_summary_is_the_reason_when_no_flags():
    v = _approve_verdict()
    v["deterministic_reasons"], v["red_flags"] = [], []
    s = api._shape(v, "r.png")
    assert s["reasons"] == [{"source": "model", "text": "Looks fine."}]


# --- HTTP endpoints (offline: validate() monkeypatched) -------------------------------------

@pytest.fixture()
def client():
    pytest.importorskip("httpx")            # FastAPI's TestClient needs httpx
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def test_health_reports_resolvable_provider(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["provider"] == "groq"


def test_validate_endpoint_returns_shaped_verdict(client, monkeypatch):
    monkeypatch.setattr(api, "validate", lambda path, provider="auto": _approve_verdict())
    r = client.post("/api/validate",
                    files={"file": ("r.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is True and body["decision"] == "approve"
    assert body["filename"] == "r.png"


def test_validate_rejects_unsupported_type(client):
    r = client.post("/api/validate",
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_validate_surfaces_missing_key_as_503(client, monkeypatch):
    def _no_key(path, provider="auto"):
        raise RuntimeError("no API key set — export GROQ_API_KEY or GEMINI_API_KEY")
    monkeypatch.setattr(api, "validate", _no_key)
    r = client.post("/api/validate",
                    files={"file": ("r.png", b"\x89PNG\r\n", "image/png")})
    assert r.status_code == 503


def test_validate_maps_all_keys_exhausted_to_429(client, monkeypatch):
    import urllib.error

    def _exhausted(path, provider="auto"):           # the provider/key chain raised its final 429
        raise urllib.error.HTTPError("u", 429, "rate limit", {}, None)

    monkeypatch.setattr(api, "validate", _exhausted)
    r = client.post("/api/validate", files={"file": ("r.png", b"\x89PNG\r\n", "image/png")})
    assert r.status_code == 429
    assert "rate-limited" in r.json()["detail"].lower()   # clean message, not a raw HTTP error
