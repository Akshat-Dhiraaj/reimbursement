"""Groq hosted-VLM extraction — read a receipt photo into a Receipt via Groq's
OpenAI-compatible API (the **API-key paradigm** for the IMAGE route).

This is the hosted counterpart to the *local* :class:`QwenVLExtractor`: the identical
Receipt-schema prompt and the SAME pure JSON->Receipt parser (`_PROMPT`,
`_parse_json_object`, `_to_receipt`, reused from `vlm_qwen` — DRY), but the model runs on
Groq instead of the local GPU. The tradeoff it exists to *measure* (the #80 scorecard):
a **light client** (no torch, no GPU — only stdlib ``urllib``) + a large hosted model,
paid for with **data egress** (the receipt image leaves the box), **rate limits**, and a
**network dependency**.

Default model is a Groq-served multimodal Llama 4 (``meta-llama/llama-4-scout-17b-16e-instruct``),
swappable via ``--model``. The key is read from ``GROQ_API_KEY`` and never stored. No heavy
deps — the request is a plain ``urllib`` POST, so importing this module is free and
``available()`` only checks for the key. (A browser User-Agent is sent because Groq sits
behind Cloudflare, which 1010-blocks default automation/datacenter client signatures —
harmless from a residential IP, necessary from a datacenter/CI one.)

Confidence note: unlike the local VLM (which reads per-token logprobs off the same greedy
pass), the chat API does not expose aligned per-digit probabilities here, so the Groq path
emits only the line-item **parse-completeness** confidence that ``_to_receipt`` derives — no
scalar-logprob signal. Honest, not hidden.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from ..models import DocumentType, Receipt
from .base import Extractor
from .vlm_qwen import _PROMPT, _parse_json_object, _to_receipt

_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
#: Groq is fronted by Cloudflare, which blocks default urllib / datacenter client
#: signatures with error 1010; a browser UA passes. Harmless from a residential IP.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".gif": "image/gif"}


def _data_url(path: str, max_side: int = 1280) -> str:
    """Read an image to a base64 data URL, downscaling large images when Pillow is present
    (a smaller payload, well under Groq's request cap); falls back to the raw file bytes."""
    mime = _MIME.get(os.path.splitext(path)[1].lower(), "image/jpeg")
    try:
        import io

        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        raw, mime = buf.getvalue(), "image/jpeg"
    except Exception:
        with open(path, "rb") as fh:
            raw = fh.read()
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


class GroqVLExtractor(Extractor):
    """Read a receipt image via Groq's hosted multimodal Llama (OpenAI-compatible API).
    Same prompt + parser as the local VLM; the variable measured is *local vs hosted API*."""

    handles = (DocumentType.IMAGE,)

    def __init__(self, model_id: str = _DEFAULT_MODEL, *, max_tokens: int = 768,
                 timeout: float = 60.0, max_retries: int = 4):
        self.model_id = model_id
        self.name = "groq:" + model_id.split("/")[-1]  # honest per-model leaderboard label
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries  # 429 backoff retries (free tier rate-limits batches)

    def available(self) -> tuple[bool, str]:
        # No heavy deps — only the key. Network/quota errors surface at extract() time as
        # an honest error count in the benchmark, not a false "unavailable".
        if not os.environ.get("GROQ_API_KEY"):
            return False, "set GROQ_API_KEY (Groq hosted VLM — note: the image leaves the box)"
        return True, ""

    def _call(self, data_url: str) -> str:
        body = json.dumps({
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }).encode("utf-8")
        req = urllib.request.Request(
            _BASE_URL, data=body, method="POST",
            headers={
                "Authorization": "Bearer " + os.environ["GROQ_API_KEY"],
                "Content-Type": "application/json",
                "User-Agent": _UA,
            },
        )
        # Retry on 429 (free-tier rate limits a batch of rapid calls) with backoff,
        # honouring a Retry-After header when present — so a benchmark isn't dominated by
        # throttling. This is itself a measured property of the API paradigm (#80).
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                return out["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.max_retries:
                    ra = e.headers.get("retry-after")
                    time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else delay)
                    delay *= 2
                    continue
                raise

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        text = self._call(_data_url(path))
        return _to_receipt(_parse_json_object(text) or {}, doc_id=doc_id or path, image_path=path)
