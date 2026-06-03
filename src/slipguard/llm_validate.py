"""Simple LLM-as-judge validity pipeline — a standalone alternative to the detector/fusion system.

**Job:** intake one reimbursement **image or PDF** → call a hosted multimodal model
(**Groq or Gemini**, selectable; key from env) with the instructions in an **external prompt file**
(`prompts/validity_prompt.md`, editable without touching code) → return the model's **structured
validity verdict** (AI-edit cues, date plausibility, arithmetic, vendor, red flags, decision).

Deliberately simple: one model call, prompt-as-config, no local detectors. PDFs go natively to
Gemini; for image-only Groq they are rasterised with pypdfium2. The prompt is honest that
pixel-level AI-edit detection is a *triage cue, not forensic proof* (see SCORECARD.md / DECISIONS.md)
— consistent with the rest of the project.

Deps (lazy-imported): Pillow (`[vlm]`) for image downscale, pypdfium2 (`[pdf]`) to rasterise PDFs for
the image-only providers. A network + key are needed for the hosted providers; LM Studio is local:
  * Groq   — `GROQ_API_KEY` (+ optional `GROQ_API_KEY_2`, `_3`, … — auto-fallback when one hits its
    rate/daily limit; OpenAI-compatible; default model meta-llama/llama-4-scout-17b-16e-instruct)
  * Gemini — `GEMINI_API_KEY` or `GOOGLE_API_KEY` (default model gemini-flash-latest; accepts PDFs directly)
  * LM Studio — a LOCAL OpenAI-compatible server (provider `lmstudio`): no API key, no quota/rate
    limit, fully private. Pass `--model` a loaded **vision** model (e.g. qwen/qwen3.5-9b); base URL via
    `LMSTUDIO_BASE_URL` (default http://localhost:1234/v1). PDFs are rasterised (image-only, like Groq).
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .extractors.vlm_qwen import _parse_json_object  # shared robust JSON-object extractor (DRY)

#: default instruction file — repo-root prompts/validity_prompt.md (editable; --prompt overrides)
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "validity_prompt.md"

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
_GEMINI_MODEL = "gemini-flash-latest"
#: LM Studio's LOCAL OpenAI-compatible server — a local vision model, no API key, no quota/rate limit.
#: Override the base with LMSTUDIO_BASE_URL; the model id comes from --model (or the LMSTUDIO_MODEL env).
_LMSTUDIO_URL = (os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
                 + "/chat/completions")
#: Groq sits behind Cloudflare, which 1010-blocks default urllib/datacenter UAs; a browser UA passes.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_IMG_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
             ".webp": "image/webp", ".gif": "image/gif"}


def load_prompt(path: Optional[str] = None) -> str:
    """Read the instruction file (default `prompts/validity_prompt.md`)."""
    return Path(path or _PROMPT_PATH).read_text(encoding="utf-8")


def _is_pdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".pdf"


def _image_part(path: str, max_side: int = 1600) -> tuple[bytes, str]:
    """An image file → (bytes, mime), downscaled if large (Pillow); falls back to raw bytes."""
    try:
        import io

        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
        buf = io.BytesIO()
        im.save(buf, "JPEG")
        return buf.getvalue(), "image/jpeg"
    except Exception:
        with open(path, "rb") as f:
            return f.read(), _IMG_MIME.get(os.path.splitext(path)[1].lower(), "image/jpeg")


def _pdf_to_pngs(path: str, max_pages: int = 2, scale: float = 2.0) -> list[tuple[bytes, str]]:
    """Render the first pages of a PDF to PNG (pypdfium2) — for image-only APIs (Groq)."""
    import io

    import pypdfium2 as pdfium
    out: list[tuple[bytes, str]] = []
    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(min(len(pdf), max_pages)):
            pil = pdf[i].render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil.save(buf, "PNG")
            out.append((buf.getvalue(), "image/png"))
    finally:
        pdf.close()
    return out


def _parts_for(path: str, provider: str) -> list[tuple[bytes, str]]:
    """Visual parts to send: a PDF goes natively to Gemini, but is rasterised for image-only Groq;
    images go as-is to both."""
    if _is_pdf(path):
        if provider == "gemini":
            with open(path, "rb") as f:
                return [(f.read(), "application/pdf")]
        return _pdf_to_pngs(path)
    return [_image_part(path)]


#: Cap any single backoff sleep. Free-tier *per-minute* throttling returns ``retry-after`` ~60s (worth
#: waiting out), but hitting a *daily* limit can return a ``retry-after`` of HOURS — without this cap
#: the client blocks for that entire time (the 85-minute "hang" a batch eval hit). Capping lets it ride
#: out a minute-window throttle yet fail fast (give up after the retries) when the wait is a daily reset.
_MAX_BACKOFF = 90.0


def _post_json(url: str, body: dict, headers: dict, timeout: float = 90.0, retries: int = 4) -> dict:
    """POST JSON with backoff on 429/503 (free-tier rate limits); each sleep is capped at
    ``_MAX_BACKOFF`` so a large ``retry-after`` (a daily limit) can't block for hours."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                ra = e.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else delay
                time.sleep(min(wait, _MAX_BACKOFF))   # cap: ride out per-minute throttling, not a daily reset
                delay *= 2
                continue
            raise


def _openai_vision_content(prompt: str, parts: list[tuple[bytes, str]]) -> list:
    """OpenAI-style multimodal user content: the prompt text + each image as a base64 data URL.
    Shared by the Groq and LM Studio callers (both speak the OpenAI chat-completions API)."""
    content: list = [{"type": "text", "text": prompt}]
    for b, mime in parts:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64," + base64.b64encode(b).decode()}})
    return content


def _numbered_keys(base: str) -> list[str]:
    """A provider's API keys in fallback order: ``BASE``, then ``BASE_2``, ``BASE_3``, … — so a
    second key automatically takes over when the first hits its rate / daily limit. Empty if none."""
    keys: list[str] = []
    primary = os.environ.get(base)
    if primary:
        keys.append(primary)
    i = 2
    while os.environ.get(f"{base}_{i}"):
        keys.append(os.environ[f"{base}_{i}"])
        i += 1
    return keys


def _call_groq(prompt: str, parts: list[tuple[bytes, str]], model: Optional[str]) -> str:
    keys = _numbered_keys("GROQ_API_KEY")
    if not keys:
        raise KeyError("GROQ_API_KEY")           # api.py maps a missing key to a clear 503
    body = {"model": model or _GROQ_MODEL, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": _openai_vision_content(prompt, parts)}]}
    last: Optional[BaseException] = None
    for n, key in enumerate(keys):
        is_last = n == len(keys) - 1
        try:
            # Rotate to the next key FAST on a rate-limited / exhausted key (retries=0); only the
            # last key absorbs the capped backoff, so a momentary all-keys-throttled minute still
            # rides out instead of giving up early.
            out = _post_json(
                _GROQ_URL, body,
                {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": _UA},
                retries=4 if is_last else 0,
            )
            return out["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and not is_last:
                last = e
                continue                          # this key is out — try the next GROQ_API_KEY_n
            raise
    raise last  # pragma: no cover - the loop returns or raises above


def _call_lmstudio(prompt: str, parts: list[tuple[bytes, str]], model: Optional[str]) -> str:
    """Call a LOCAL vision model via LM Studio's OpenAI-compatible server. No API key and no
    rate/quota limits — but local inference is slower, so the timeout is generous and we don't retry."""
    mdl = model or os.environ.get("LMSTUDIO_MODEL")
    if not mdl:
        raise RuntimeError("LM Studio needs a model id — pass --model (a vision model loaded in LM "
                           "Studio, e.g. qwen/qwen3.5-9b) or set LMSTUDIO_MODEL")
    # Local reasoning/hybrid models (Qwen3, Gemma) emit hidden reasoning BEFORE the answer, so the
    # token budget must cover reasoning + the JSON or `content` comes back empty — hence 4096, not 1024.
    body = {"model": mdl, "temperature": 0, "max_tokens": 4096,
            "messages": [{"role": "user", "content": _openai_vision_content(prompt, parts)}]}
    out = _post_json(_LMSTUDIO_URL, body,
                     {"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
                     timeout=600.0, retries=1)
    return out["choices"][0]["message"]["content"]


def _call_gemini(prompt: str, parts: list[tuple[bytes, str]], model: Optional[str]) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    gparts: list = [{"text": prompt}]
    for b, mime in parts:
        gparts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(b).decode()}})
    body = {"contents": [{"parts": gparts}],
            "generationConfig": {"temperature": 0, "response_mime_type": "application/json"}}
    # Auth via the X-goog-api-key header (works for both classic AIza and newer AQ. keys),
    # rather than the ?key= query param — matches Google's current docs.
    url = _GEMINI_URL.format(m=model or _GEMINI_MODEL)
    out = _post_json(url, body, {"Content-Type": "application/json", "X-goog-api-key": key})
    return out["candidates"][0]["content"]["parts"][0]["text"]


_DOTENV_LOADED = False


def load_local_env() -> None:
    """Best-effort: populate ``os.environ`` from a repo-root ``.env`` for keys not already set, so a
    local API key works without exporting it. Idempotent (loads at most once) and never overrides an
    existing variable. Called from the CLI entry point and the web backend — deliberately NOT from
    :func:`resolve_provider`, so library/test calls keep their explicit-environment semantics."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.is_file():
        return
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


def resolve_provider(provider: str = "auto") -> str:
    """Pick the provider: explicit name, else Gemini if its key is set, else Groq."""
    if provider and provider != "auto":
        return provider
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    raise RuntimeError("no API key set — export GROQ_API_KEY or GEMINI_API_KEY / GOOGLE_API_KEY")


# --- deterministic cross-check (the refinement: don't trust the LLM's self-judged math) ------

_SEVERITY = {"approve": 0, "review": 1, "reject": 2}


def _worst(a: str, b: str) -> str:
    """The stricter of two decisions (approve < review < reject)."""
    return a if _SEVERITY.get(a, 1) >= _SEVERITY.get(b, 1) else b


def _f(x) -> Optional[float]:
    """A money value -> float (shared parser for strings); None / non-numeric -> None."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    from .money import parse_money
    return parse_money(str(x))


def _verdict_to_receipt(verdict: dict, path: str):
    """Build a Receipt from the LLM verdict's extracted fields so the deterministic detectors can
    re-check the model's own numbers / date / tax-id (and the real source file)."""
    from .extractors.vlm_qwen import _parse_date
    from .models import DocumentType, Receipt
    from .routing import route_path
    route = route_path(path)
    return Receipt(
        doc_id=path,
        vendor_name=(verdict.get("vendor") or "(unknown)"),
        date=_parse_date(verdict.get("date")),
        currency=(verdict.get("currency") or "USD"),
        country=(verdict.get("country") or "US"),
        vendor_tax_id=(verdict.get("tax_id") or None),
        subtotal=_f(verdict.get("subtotal")), tax_amount=_f(verdict.get("tax")),
        total=_f(verdict.get("total")),
        service_charge=_f(verdict.get("service_charge")), discount=_f(verdict.get("discount")),
        source=route, source_path=path,
        image_path=path if route is DocumentType.IMAGE else None,
    )


def reconcile(verdict: dict, path: str) -> dict:
    """Cross-check the LLM verdict with the pure-Python detectors run on the model's OWN extracted
    fields (and the real file): the final decision is the STRICTER of the LLM's and the
    deterministic layer's — so a confident LLM `approve` is overruled to review/reject when the
    arithmetic doesn't reconcile, the date is impossible, or a tax-id fails its checksum. It NEVER
    relaxes the LLM decision. Dependency-free (heavy provenance detectors abstain if their extras
    are absent); this patches the measured 'confident arithmetic misread' weakness for free."""
    from .detectors import default_detectors
    from .fusion import Fuser
    det = Fuser().verdict(path, [d.run(_verdict_to_receipt(verdict, path)) for d in default_detectors()])
    llm_decision = verdict.get("decision", "review")
    verdict["llm_decision"] = llm_decision
    verdict["deterministic_decision"] = det.decision.value
    verdict["deterministic_reasons"] = list(det.reasons)
    verdict["decision"] = _worst(llm_decision, det.decision.value)
    return verdict


def validate(path: str, *, provider: str = "auto", prompt_path: Optional[str] = None,
             model: Optional[str] = None, cross_check: bool = True) -> dict:
    """Intake one image/PDF, call the chosen API with the external prompt, return the verdict dict.
    Adds `_provider` / `_path` for traceability, fails safe to `decision="review"` if the model
    didn't return one, and keeps a `_raw` snippet when the reply wasn't parseable JSON. With
    ``cross_check`` (default), :func:`reconcile` overrules the decision to the stricter of the LLM's
    and the deterministic detectors' (run on the model's own numbers) — never relaxing it."""
    prompt = load_prompt(prompt_path)
    prov = resolve_provider(provider)
    parts = _parts_for(path, prov)
    caller = {"gemini": _call_gemini, "lmstudio": _call_lmstudio}.get(prov, _call_groq)
    text = caller(prompt, parts, model)

    verdict = _parse_json_object(text) or {}
    if "decision" not in verdict:
        verdict["decision"] = "review"   # fail safe: unparseable / no decision → human looks
        verdict["_raw"] = (text or "")[:500]
    verdict["_provider"], verdict["_path"] = prov, path
    if cross_check:
        verdict = reconcile(verdict, path)
    return verdict
