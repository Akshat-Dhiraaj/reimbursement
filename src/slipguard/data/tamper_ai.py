"""AI-generated fake receipts — the heavier `make-fakes` methods.

* ``make_gemini`` — edit real receipts with Google's **Gemini image model ("Nano Banana",
  default ``gemini-2.5-flash-image``)** via the API. Realistic, seamless edits — but image
  generation has a *very* tight free-tier quota (it 429s when spent; wait for reset or use a paid
  tier). Needs ``GEMINI_API_KEY``; the receipt leaves the box.
* ``make_local`` — edit receipts with a **local open diffusion model** (e.g. FLUX.1-schnell,
  Apache-2.0) via ``diffusers``. Fully local/private, but needs a real GPU + a multi-GB model
  download, so it will not run on a no-GPU machine. Left as an explicit, confirm-first setup.

The free/portable method (pure-Python Pillow overlay) lives in ``tamper.py``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from ._common import list_source_images

_GEMINI_IMG_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
_DEFAULT_GEMINI_IMG = "gemini-2.5-flash-image"   # "Nano Banana" (GA); override with --model

#: edit name -> instruction. Each turns a genuine receipt into a specific fraud type.
_EDITS = {
    "inflated_total": "Edit this receipt image: change the printed grand total so it reads "
                      "$9,999.99. Keep the layout, fonts, logo, line items and every other detail "
                      "identical so it still looks like a genuine receipt. Return only the edited image.",
    "future_date": "Edit this receipt image: change the transaction date to 31 December 2099. "
                   "Keep everything else identical. Return only the edited image.",
}
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _image_from_response(out: dict) -> Optional[bytes]:
    """Pull the first inline image out of a Gemini generateContent response (camelCase or snake)."""
    for cand in out.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            d = part.get("inlineData") or part.get("inline_data")
            if d and d.get("data"):
                return base64.b64decode(d["data"])
    return None


def make_gemini(src_dir: str, out_dir: str, limit: Optional[int] = None,
                model: Optional[str] = None) -> list[Path]:
    """Generate AI-edited fakes via the Gemini image model ("Nano Banana"). Raises a clear error if
    the free-tier image quota is exhausted (429) so it fails fast rather than hammering the API."""
    import os
    import urllib.error

    from ..llm_validate import _post_json
    model = model or _DEFAULT_GEMINI_IMG
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set — needed for the Nano Banana image edits")

    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srcs = list_source_images(src, limit)

    url = _GEMINI_IMG_URL.format(m=model)
    headers = {"Content-Type": "application/json", "X-goog-api-key": key}
    made: list[Path] = []
    for p in srcs:
        data = base64.b64encode(p.read_bytes()).decode()
        mime = _MIME.get(p.suffix.lower(), "image/jpeg")
        for name, instr in _EDITS.items():
            body = {"contents": [{"parts": [
                {"text": instr}, {"inline_data": {"mime_type": mime, "data": data}}]}]}
            try:
                resp = _post_json(url, body, headers, timeout=120.0, retries=0)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise RuntimeError(
                        f"Gemini image quota exhausted (429) on {model}. Free-tier image generation "
                        f"is very limited - wait for the daily reset or use a paid tier. "
                        f"({len(made)} fake(s) written so far.)"
                    )
                raise
            img = _image_from_response(resp)
            if img:
                dst = out / f"{p.stem}__{name}.png"
                dst.write_bytes(img)
                made.append(dst)
    return made


def make_local(src_dir: str, out_dir: str, limit: Optional[int] = None,
               model: Optional[str] = None) -> list[Path]:
    """Local diffusion edits — intentionally not auto-installed (needs a GPU + a multi-GB model)."""
    raise SystemExit(
        "--method local is not set up yet. It needs a local diffusion stack (diffusers + torch +\n"
        "a COMMERCIAL-SAFE model such as FLUX.1-schnell, Apache-2.0) and a real GPU - a heavy,\n"
        "multi-GB install that won't run on a no-GPU laptop. Confirm and I'll wire it up.\n"
        "For free/portable fakes use:  slipguard make-fakes --method pytamper"
    )
