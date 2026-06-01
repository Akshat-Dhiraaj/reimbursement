"""VLM extraction via Qwen2.5-VL — read a receipt photo end-to-end into a Receipt.

This is the first *real* extractor for the IMAGE route (the audit named faithful
extraction as the binding constraint). It implements the same `Extractor` contract as
everything else, so `eval-extract` ranks it on field accuracy against the WildReceipt
oracle with no special-casing — we pick the extractor by numbers, not reputation.

Why a Qwen-VL checkpoint: strong at reading messy receipts end-to-end (no separate
OCR+layout stage) with a **commercial-safe** default. Licence matters per checkpoint
(verified against the HF metadata — see DECISIONS.md): Qwen2.5-VL-**7B**-Instruct and
Qwen2-VL-**2B**-Instruct are Apache-2.0, but Qwen2.5-VL-**3B** ships with **no clear
commercial licence**, so we do *not* use it. The 8 GB dev GPU fits the 2B natively, so
that is the default; the Apache-2.0 7B is selectable via ``--model`` and runs with CPU
offload. Loading goes through the transformers **Auto** classes, so any HF
vision-language checkpoint is a swappable candidate — the checkpoint (and its
licence/size) is just a parameter.

The heavy deps (torch / transformers / Pillow) are **imported lazily inside the
methods**, so importing this module — and the whole package — stays dependency-free;
`available()` reports missing deps without loading the model, letting the benchmark skip
an un-runnable candidate cleanly.

Confidence note: a VLM does not emit calibrated per-field confidence, so we do **not**
fabricate `field_confidence` here — the benchmark measures raw field accuracy, and
confidence calibration (to arm the `arithmetic` abstain guard) is tracked as later work.
"""

from __future__ import annotations

import json
import re
from datetime import date as Date
from datetime import datetime
from typing import Any, Optional

from ..models import DocumentType, LineItem, Receipt
from ..money import parse_money
from .base import Extractor

#: default checkpoint — Apache-2.0 AND fits an 8 GB GPU natively. (Qwen2.5-VL-3B has no
#: clear commercial licence; the Apache-2.0 7B fits only with CPU offload.) See DECISIONS.md.
_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

_PROMPT = (
    "You are an expert receipt parser. Read this receipt image and extract its fields. "
    "Respond with ONLY a single JSON object — no prose, no markdown, no code fences. "
    "Schema:\n"
    "{\n"
    '  "vendor_name": string,        // store / merchant name\n'
    '  "date": "YYYY-MM-DD" or null, // transaction date\n'
    '  "currency": string,           // ISO code, e.g. "USD"\n'
    '  "line_items": [ {"description": string, "quantity": number, '
    '"unit_price": number, "amount": number} ],\n'
    '  "subtotal": number or null,\n'
    '  "tax_amount": number or null,\n'
    '  "total": number or null\n'
    "}\n"
    "Numbers must be plain: no currency symbols, no thousands separators. "
    "Use null for any field not present (and [] for line_items). Output JSON only."
)

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y")


def _num(x: Any) -> Optional[float]:
    """Coerce a model-produced money value to float. Numerics pass through (a JSON
    number is already clean); strings go through the shared US/EU-aware parser so a
    stray symbol or a ``1.234,56`` slipping past the prompt doesn't 100x the amount."""
    if x is None:
        return None
    if isinstance(x, bool):  # avoid True/False -> 1.0/0.0
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return parse_money(str(x))


def _parse_date(x: Any) -> Optional[Date]:
    if not x or not isinstance(x, str):
        return None
    s = x.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_json_object(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response, ignoring code fences and any
    surrounding prose. Pure + model-free so it can be unit-tested directly."""
    s = text.strip()
    if "```" in s:  # drop ```json ... ``` fencing if the model added it
        s = re.sub(r"```(?:json)?", "", s).strip()
    start = s.find("{")
    if start == -1:
        return None
    try:  # raw_decode parses one value at `start` and ignores any trailing text
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _to_receipt(data: dict, doc_id: str, image_path: str) -> Receipt:
    """Map the parsed JSON dict to a Receipt, coercing types defensively. Pure +
    model-free (unit-tested). A field the model omitted stays None / empty."""
    items: list[LineItem] = []
    for it in data.get("line_items") or []:
        if not isinstance(it, dict):
            continue
        amount = _num(it.get("amount"))
        if amount is None:  # an item with no money value is noise, skip it
            continue
        items.append(LineItem(
            description=str(it.get("description", "") or ""),
            quantity=_num(it.get("quantity")) or 1.0,
            unit_price=_num(it.get("unit_price")) if _num(it.get("unit_price")) is not None else amount,
            amount=amount,
        ))

    vendor = data.get("vendor_name")
    return Receipt(
        doc_id=doc_id,
        vendor_name=str(vendor).strip() if vendor else "(unknown)",
        date=_parse_date(data.get("date")),
        currency=str(data.get("currency") or "USD"),
        country="US",
        line_items=items,
        subtotal=_num(data.get("subtotal")),
        tax_amount=_num(data.get("tax_amount")),
        total=_num(data.get("total")),
        source=DocumentType.IMAGE,
        source_path=image_path,
        image_path=image_path,
    )


class QwenVLExtractor(Extractor):
    """Prompt a Qwen-VL model to emit the Receipt schema as JSON, then parse it.
    Model-agnostic via the transformers Auto classes, so the checkpoint is swappable."""

    handles = (DocumentType.IMAGE,)

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        max_new_tokens: int = 768,
        max_image_side: int = 1280,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.name = model_id.split("/")[-1]  # honest per-checkpoint leaderboard label
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._processor = None

    def available(self) -> tuple[bool, str]:
        # Probe with find_spec / metadata only — never import torch/transformers here,
        # so this stays fast (the unit-test suite calls it) and the model never loads.
        missing = [m for m in ("torch", "transformers", "PIL") if not _importable(m)]
        if missing:
            return False, f"missing deps: {', '.join(missing)} — pip install -e \".[vlm]\""
        ver = _pkg_version("transformers")
        if ver is not None and ver < (4, 45):
            dotted = ".".join(map(str, ver))
            return False, f"transformers {dotted} < 4.45 (Qwen-VL needs >= 4.45)"
        return True, ""

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:  # the generic VL class dispatches to Qwen2-VL / Qwen2.5-VL by config
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM

        dtype = getattr(torch, self.dtype, torch.bfloat16)
        try:  # transformers 5.x prefers `dtype=`; older wants `torch_dtype=`
            self._model = _AutoVLM.from_pretrained(
                self.model_id, dtype=dtype, device_map=self.device_map
            )
        except TypeError:
            self._model = _AutoVLM.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map=self.device_map
            )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _resize(self, image):
        w, h = image.size
        longest = max(w, h)
        if longest <= self.max_image_side:
            return image
        scale = self.max_image_side / longest
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        import torch
        from PIL import Image

        self._ensure_model()
        image = self._resize(Image.open(path).convert("RGB"))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": _PROMPT},
        ]}]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        text_out = self._processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        data = _parse_json_object(text_out) or {}
        return _to_receipt(data, doc_id=doc_id or path, image_path=path)


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _pkg_version(name: str) -> Optional[tuple[int, ...]]:
    """(major, minor) of an installed package via metadata — no import of the package."""
    try:
        from importlib.metadata import version

        return tuple(int(p) for p in version(name).split(".")[:2])
    except Exception:
        return None
