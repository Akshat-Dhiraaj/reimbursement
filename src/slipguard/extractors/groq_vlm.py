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
import os
from typing import Optional

# Reuse the validate pipeline's HTTP plumbing — one POST-with-backoff, one image loader, one
# Cloudflare-passing browser UA — so this hosted-Groq path can't drift from it and inherits the
# _MAX_BACKOFF cap + safe retry-after parsing for free (DRY).
from ..llm_validate import _GROQ_URL, _UA, _image_part, _post_json
from ..models import DocumentType, Receipt
from .base import Extractor
from .vlm_qwen import _PROMPT, _parse_json_object, _to_receipt

_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _data_url(path: str, max_side: int = 1280) -> str:
    """Read an image to a base64 data URL, downscaling large images when Pillow is present
    (a smaller payload, well under Groq's request cap); falls back to the raw file bytes."""
    raw, mime = _image_part(path, max_side)
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
        # Shared POST-with-backoff (429/503, capped) — the same plumbing the validate pipeline
        # uses, so the free-tier throttling behaviour can't diverge between the two Groq callers.
        body = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }
        out = _post_json(
            _GROQ_URL, body,
            {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"],
             "Content-Type": "application/json", "User-Agent": _UA},
            timeout=self.timeout, retries=self.max_retries,
        )
        return out["choices"][0]["message"]["content"]

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        text = self._call(_data_url(path))
        return _to_receipt(_parse_json_object(text) or {}, doc_id=doc_id or path, image_path=path)
