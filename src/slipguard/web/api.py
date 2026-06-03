"""FastAPI backend for the slipguard web UI.

One job: accept an uploaded reimbursement **image or PDF**, run it through the existing
:func:`slipguard.llm_validate.validate` pipeline (a hosted multimodal model judge + the
deterministic arithmetic/checksum cross-check), and return a verdict the React frontend can render
as **Approved / Not approved — with the reasons in either case**.

Endpoints
---------
``GET  /api/health``   — which provider (Groq/Gemini) is resolvable from the environment.
``POST /api/validate`` — multipart upload (``file`` + optional ``provider``) → shaped verdict JSON.

Design
------
* No new logic — this is a transport over ``validate()``. The fraud-decision rules stay in one place.
* ``validate()`` does blocking network I/O, so it runs in a threadpool (keeps the event loop free).
* The shaped response surfaces **both** the model's reasons (``summary`` / ``red_flags``) and the
  deterministic cross-check's reasons (``deterministic_reasons``), tagged by source — so the UI can
  show *why* a clean receipt was approved and *why* a suspicious one was held, which is the two-layer
  design made visible.
* If ``frontend/dist`` has been built, it is served at ``/`` so the whole app runs from one port.

Dependencies (the ``[web]`` extra): fastapi (MIT), uvicorn (BSD-3-Clause), python-multipart
(Apache-2.0) — all commercial-safe. An API key (``GROQ_API_KEY`` or ``GEMINI_API_KEY`` /
``GOOGLE_API_KEY``) is read from the environment or a repo-root ``.env``.
"""

from __future__ import annotations

import os
import tempfile
import urllib.error
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised only without the [web] extra
    raise ImportError(
        "The slipguard web UI needs FastAPI. Install the web extra:\n"
        '    pip install -e ".[web]"'
    ) from exc

from ..llm_validate import load_local_env, resolve_provider, validate

#: Upload types we accept — the same set the validate pipeline can read (images + PDF).
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".pdf"}
#: Reject obviously-too-large uploads before touching disk / the API (phone photos are < ~15 MB).
_MAX_BYTES = 25 * 1024 * 1024
#: Vite dev server origins — the frontend talks to this API cross-origin during `npm run dev`.
_DEV_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",  # `vite preview`
]
#: Built frontend (optional) — if present, serve it so the whole app runs from one uvicorn port.
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


load_local_env()  # pick up GROQ_API_KEY / GEMINI_API_KEY from a repo-root .env (shared with the CLI)

app = FastAPI(
    title="slipguard",
    description="Reimbursement receipt / invoice validity check (LLM judge + deterministic cross-check).",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _num(x) -> Optional[float]:
    """A value that should be a number → float, else None (confidence sometimes arrives as a string)."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _shape(verdict: dict, filename: Optional[str]) -> dict:
    """Map the raw ``validate()`` verdict to the response the frontend renders.

    ``reasons`` is the union the UI shows in both the approve and not-approve cases, each tagged by
    ``source`` ("deterministic" = the pure-Python re-check of the model's own numbers; "model" = the
    vision model's stated concerns). On an approval the deterministic reasons read positively
    (e.g. "all arithmetic reconciles"), which is exactly the "show the reason" the user asked for."""
    decision = verdict.get("decision", "review")
    reasons: list[dict] = []
    for text in verdict.get("deterministic_reasons", []) or []:
        reasons.append({"source": "deterministic", "text": str(text)})
    for text in verdict.get("red_flags", []) or []:
        reasons.append({"source": "model", "text": str(text)})
    if not reasons and verdict.get("summary"):
        # A clean approval with no itemised reasons — still show the model's one-liner as the reason.
        reasons.append({"source": "model", "text": str(verdict["summary"])})

    return {
        "approved": decision == "approve",
        "decision": decision,                       # approve | review | reject (reconciled final)
        "confidence": _num(verdict.get("confidence")),
        "summary": verdict.get("summary") or "",
        "reasons": reasons,
        "llm_decision": verdict.get("llm_decision", decision),
        "deterministic_decision": verdict.get("deterministic_decision"),
        "ai_or_edit_signs": list(verdict.get("ai_or_edit_signs") or []),
        "fields": {
            "vendor": verdict.get("vendor"),
            "date": verdict.get("date"),
            "currency": verdict.get("currency"),
            "subtotal": verdict.get("subtotal"),
            "tax": verdict.get("tax"),
            "service_charge": verdict.get("service_charge"),
            "discount": verdict.get("discount"),
            "total": verdict.get("total"),
            "tax_id": verdict.get("tax_id"),
            "country": verdict.get("country"),
        },
        "checks": {
            "ai_or_edit_suspected": verdict.get("ai_or_edit_suspected"),
            "date_valid": verdict.get("date_valid"),
            "arithmetic_consistent": verdict.get("arithmetic_consistent"),
        },
        "provider": verdict.get("_provider"),
        "filename": filename,
        "verdict": verdict,                          # full raw verdict, for the curious / debugging
    }


@app.get("/api/health")
def health() -> dict:
    """Is the service ready to validate? Reports the resolvable provider and which keys are present
    (names only — never values)."""
    keys = {
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    }
    try:
        return {"ok": True, "provider": resolve_provider("auto"), "keys": keys}
    except RuntimeError as exc:
        return {"ok": False, "provider": None, "keys": keys, "detail": str(exc)}


@app.post("/api/validate")
async def api_validate(
    file: UploadFile = File(...),
    provider: str = Form("auto"),
) -> JSONResponse:
    """Validate one uploaded receipt/invoice. ``provider`` is ``auto`` | ``groq`` | ``gemini``."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or '(none)'}'. "
                   f"Accepted: {', '.join(sorted(_ALLOWED_SUFFIXES))}.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {_MAX_BYTES // (1024*1024)} MB).")

    # validate() reads from a path; stage the upload to a temp file (manual unlink — Windows can't
    # reopen a still-open NamedTemporaryFile).
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        verdict = await run_in_threadpool(lambda: validate(tmp, provider=provider))
    except RuntimeError as exc:                      # no API key configured
        raise HTTPException(status_code=503, detail=str(exc))
    except KeyError as exc:                           # provider chosen but its key absent
        raise HTTPException(status_code=503, detail=f"Missing API key for the selected provider: {exc}.")
    except ImportError as exc:                        # e.g. PDF upload but pypdfium2 not installed
        raise HTTPException(status_code=501, detail=f"A dependency is missing for this input: {exc}.")
    except urllib.error.HTTPError as exc:             # every key/provider exhausted (chain raised the last 429)
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="All configured API keys are rate-limited or out of daily quota. "
                       "Add another key (e.g. GROQ_API_KEY_2) or try again shortly.",
            )
        raise HTTPException(status_code=502, detail=f"Provider returned HTTP {exc.code}.")
    except Exception as exc:                          # network / model / parse failure
        raise HTTPException(status_code=502, detail=f"Validation failed: {exc}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    return JSONResponse(_shape(verdict, file.filename))


# Serve the built frontend last (so /api/* always wins). Only mounted if `npm run build` has run.
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
