"""VLM extraction via a Qwen-VL checkpoint — read a receipt photo end-to-end into a Receipt.

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

Confidence note: a VLM does **not** emit a calibrated per-field probability, so we do
not fabricate one. We report two *observed* signals into ``Receipt.field_confidence``,
both of which arm the `arithmetic` abstain guard, and both honest about their limits:

1. **Parse completeness** (line items): when the model emits line items we cannot all
   parse (missing/garbled amounts), the captured line-item sum is unreliable, so
   ``field_confidence["line_items"]`` carries the fraction we parsed. That covers exactly
   the under-capture case the FP audit named — a low ratio makes arithmetic abstain instead
   of crying fraud on a ``subtotal != sum(items)`` gap that is really a capture artifact.
   Its blind spot: it sees *emitted-but-unparseable* loss only, never a cleanly-parsed-but-
   **mislabeled scalar** (the head-to-head showed those, not capture loss, drive the FP).

2. **Token logprobs on the scalar money fields**: the same greedy pass that produces the
   value also exposes, per emitted token, the model's own probability for that token. We
   align those probabilities back to the digits of ``subtotal`` / ``tax_amount`` / ``total``
   and record the **least-confident digit's** probability per field (see
   ``_field_confidence_from_tokens``). A value the model emitted hesitantly gets a low
   probability → arithmetic abstains on that *misread* scalar — the gap (1) cannot see — and
   it rides the SAME forward pass, so it is essentially free (no extra inference, unlike
   re-sampling). Its blind spot, stated not hidden: a logprob measures the model's
   *self-assurance*, not *truth* — a confidently-wrong read (a stable misread) can still
   score high, and a field the model simply omits has no token to score at all (a coverage
   gap, orthogonal to confidence). The **value** is always the deterministic greedy read, so
   measured field accuracy is unchanged; only the confidence annotation is added.

Both record only sub-1.0 values, so a clean / confident extraction leaves
``field_confidence`` empty (== fully trusted) and behaviour is unchanged unless we actually
observe parse loss or a low-probability digit. Both ride the single greedy decode — no extra
inference cost.
"""

from __future__ import annotations

import json
import re
from datetime import date as Date
from datetime import datetime
from typing import Any, Optional

from ..models import DocumentType, LineItem, Receipt
from ..money import parse_money
from .base import Extractor, importable

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
    model-free (unit-tested). A field the model omitted stays None / empty.

    Also records line-item parse completeness in ``field_confidence`` (see the module
    "Confidence note"): if the model emitted items we could not all parse, the kept
    fraction is stored under ``"line_items"`` so the `arithmetic` guard can abstain on
    a capture artifact instead of crying fraud. Only sub-1.0 values are recorded, so a
    clean extraction leaves ``field_confidence`` empty == fully trusted (no behaviour
    change)."""
    items: list[LineItem] = []
    emitted = 0
    for it in data.get("line_items") or []:
        emitted += 1  # count every entry the model put in line_items, parseable or not
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

    # Parse-completeness confidence (not a calibrated probability): the fraction of the
    # emitted line items we could actually use. Recorded only when < 1.0, so a clean
    # extraction keeps field_confidence empty (== fully trusted) and behaviour is
    # unchanged. A low ratio arms the arithmetic abstain guard on under-capture.
    field_confidence: dict[str, float] = {}
    if emitted and len(items) < emitted:
        field_confidence["line_items"] = round(len(items) / emitted, 3)

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
        field_confidence=field_confidence,
    )


#: scalar money fields whose token-logprob confidence arms the `arithmetic` guard
#: (the keys it reads in ``Receipt.field_confidence``). ``date`` is deliberately omitted:
#: no detector consumes a date confidence yet, so emitting one would be a dead signal.
_CONF_FIELDS = ("subtotal", "tax_amount", "total")

def _value_re(field: str) -> "re.Pattern[str]":
    """Match a JSON scalar as the model emits it — ``"total": 58.22`` / ``"total": null``
    and, defensively, a quoted number (``"total": "58.22"``) the prompt told it not to
    produce. Capture group 1 is the value's digits (or the literal ``null``); the optional
    leading quote sits outside the group so the char span we score is the number itself."""
    return re.compile(r'"' + re.escape(field) + r'"\s*:\s*"?(null|-?\d[\d.,]*)')


def _incremental_spans(decode, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Decode ``token_ids`` one prefix at a time and return ``(text, spans)`` where
    ``spans[i]`` is the ``[start, end)`` character range token ``i`` contributed to ``text``.
    Incremental prefix decoding is the robust way to map subword tokens back to characters:
    the i-th token occupies whatever ``decode(ids[:i+1])`` appended past ``decode(ids[:i])``.
    A token that adds no visible text (e.g. a skipped special token) gets an empty span and
    so covers no field value. ``decode`` is injected — a plain ``list[int] -> str`` callable —
    so this is pure + GPU-free (the tests pass a trivial char decoder; production passes the
    HF tokenizer's ``decode``)."""
    spans: list[tuple[int, int]] = []
    prev = ""
    for i in range(len(token_ids)):
        cur = decode(token_ids[: i + 1])
        start = len(prev)
        spans.append((start, max(start, len(cur))))
        prev = cur
    return prev, spans


def _field_confidence_from_tokens(
    text: str, spans: list[tuple[int, int]], probs: list[float], fields=_CONF_FIELDS
) -> dict[str, float]:
    """Map per-token probabilities onto the scalar money fields: for each present, non-null
    field, find its numeric value in the JSON ``text``, take the tokens whose char spans
    cover that value, and record the **min** of their probabilities — the least-confident
    digit. Min (not mean or product) because one shaky digit should drag the whole field's
    confidence down, and min is length-unbiased: a long amount is not penalised merely for
    spanning more tokens. A field that is absent or explicitly ``null`` is skipped (no value
    to score), and only sub-1.0 ratios are recorded — mirroring the parse-completeness
    convention so a fully confident read leaves ``field_confidence`` empty (== trusted) and
    the arithmetic guard behaves exactly as before. Pure + GPU-free (``spans``/``probs`` come
    from :func:`_incremental_spans` and ``compute_transition_scores`` in production)."""
    conf: dict[str, float] = {}
    for fld in fields:
        m = _value_re(fld).search(text)
        if m is None or m.group(1) == "null":
            continue  # field not emitted, or emitted as null -> no digits to be (un)sure of
        vstart, vend = m.span(1)
        covering = [p for (ts, te), p in zip(spans, probs)
                    if te > ts and ts < vend and te > vstart]  # tokens overlapping the value
        if not covering:
            continue
        ratio = round(min(covering), 3)
        if ratio < 1.0:  # record only uncertainty (mirrors the line_items convention)
            conf[fld] = ratio
    return conf


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
        missing = [m for m in ("torch", "transformers", "PIL") if not importable(m)]
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

    def _read_with_confidence(self, inputs) -> tuple[str, dict[str, float]]:
        """One greedy decode that returns both the reply text and a per-scalar token-logprob
        confidence. ``output_scores`` keeps the per-step logits; ``compute_transition_scores``
        turns them into the log-probability the model assigned to *each token it actually
        chose*, which ``.exp()`` makes a probability. We decode the generated ids
        incrementally to map every token to its character span, then
        :func:`_field_confidence_from_tokens` aligns those probabilities to the money values.
        The text is taken from the SAME incremental decode the spans index into, so the JSON
        we parse and the spans we score can never drift apart."""
        import torch

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
        token_ids = gen_ids[0].tolist()
        # log-prob of each chosen token -> probability (one value per generated token)
        scores = self._model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )
        probs = scores[0].exp().tolist()

        tok = getattr(self._processor, "tokenizer", self._processor)
        text, spans = _incremental_spans(
            lambda ids: tok.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            token_ids,
        )
        return text, _field_confidence_from_tokens(text, spans, probs)

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
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

        # One greedy decode is the value source AND the confidence source: deterministic, so
        # the extracted fields (and measured accuracy) are unchanged; the token logprobs that
        # annotate confidence come free from the same pass — no extra inference.
        text, token_conf = self._read_with_confidence(inputs)
        receipt = _to_receipt(_parse_json_object(text) or {}, doc_id=doc_id or path, image_path=path)
        receipt.field_confidence.update(token_conf)  # scalar logprob conf rides alongside line_items
        return receipt


def _pkg_version(name: str) -> Optional[tuple[int, ...]]:
    """(major, minor) of an installed package via metadata — no import of the package."""
    try:
        from importlib.metadata import version

        return tuple(int(p) for p in version(name).split(".")[:2])
    except Exception:
        return None
