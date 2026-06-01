"""OCR + heuristic KIE extraction via docTR — the second IMAGE-route extractor.

Why this exists: the project picks extractors by *measured* field accuracy, not
reputation, so the VLM (:mod:`slipguard.extractors.vlm_qwen`) needs at least one
head-to-head rival on the same ``eval-extract`` oracle. This is that rival, built from a
different paradigm — a two-stage OCR pipeline (text detection + recognition) followed by
a small, transparent key-information-extraction (KIE) layer that maps the read lines onto
the Receipt schema with position + keyword heuristics. No second ML model, no
fine-tuning, fully inspectable; the leaderboard says how far heuristics get versus the
VLM's end-to-end reading.

Licence (commercial-safe): docTR is Apache-2.0 and its default pretrained detection /
recognition checkpoints are Apache-2.0, and it runs on the torch the ``[vlm]`` extra
already pulls in. Heavy imports (doctr / torch) are **lazy inside the methods**, so
importing this module — and the whole package — stays dependency-free; ``available()``
reports missing deps via ``find_spec`` without loading anything.

Confidence: unlike a VLM, an OCR recogniser emits a real per-word recognition confidence
(softmax over the character decoder). We surface it honestly — each money field records
the recognition confidence of the line it was read from under
:attr:`slipguard.models.Receipt.field_confidence` (keys ``subtotal`` / ``tax_amount`` /
``total``, and the mean over the item lines for ``line_items``), so a garbled read makes
the ``arithmetic`` detector abstain instead of crying fraud — the same guard the VLM arms
via parse-completeness, here driven by a genuine OCR confidence. Honest limit: this is a
*character-recognition* confidence (did we read the glyphs right), not a field-*labelling*
confidence (did we pick the right line as "total") — it catches a garbled read, not a
confident misread of the wrong row.

KIE heuristics (deliberately simple and transparent): we first **row-merge** docTR's
output (:func:`_merge_rows`) because docTR emits each row's left-column label and
right-column amount as separate lines at the same height — without this the keyword and
its amount never share a line. Then: vendor = the most letter-heavy line in the top band
of the page; date = the first parseable date anywhere; subtotal / tax / total = the
right-most money value on the bottom-most row carrying the matching keyword; line items =
non-summary rows that end in a price-shaped amount.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

from ..models import DocumentType, LineItem, Receipt
from ..money import parse_money
from .base import Extractor

# --- KIE keyword vocabulary --------------------------------------------------

_SUBTOTAL_KW = ("subtotal", "sub total", "sub-total")
_TAX_KW = ("tax", "gst", "vat", "hst", "pst", "cgst", "sgst", "igst", "service charge")
#: "total" is matched only after subtotal lines are excluded (note that the string
#: "subtotal" itself contains "total", so a naive match would pick the subtotal row).
_TOTAL_KW = ("grand total", "total", "amount due", "balance due", "total due", "amount payable")
#: lines that are summary / payment rows and must never be harvested as purchasable items.
_NONITEM_KW = _SUBTOTAL_KW + _TAX_KW + _TOTAL_KW + (
    "change", "cash", "tender", "card", "visa", "master", "amex", "debit", "credit",
    "balance", "due", "payment", "paid", "round", "rounding", "qty", "quantity",
    "amount", "discount", "savings",
)

#: a money-like run: a digit-led group (which may carry ``,`` / ``.`` grouping) or a
#: lone digit. We take the *right-most* one on a line (amounts sit in the right column).
_MONEY_TOKEN_RE = re.compile(r"-?\d[\d.,]*\d|-?\d")
#: "looks like a printed price": ends in a separator + exactly two decimals. Screens out
#: phone numbers, quantities and dates when harvesting line-item prices.
_PRICE_SHAPE_RE = re.compile(r"[.,]\d{2}$")


# --- date finding ------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    start=1,
)}
_DATE_NUMERIC_YMD = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_DATE_NUMERIC_DMY = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b")
_DATE_D_MON_Y = re.compile(r"\b(\d{1,2})[ .\-]*([A-Za-z]{3,9})[ .\-,]*(\d{2,4})\b")
_DATE_MON_D_Y = re.compile(r"\b([A-Za-z]{3,9})[ .\-]*(\d{1,2})[ .\-,]*(\d{2,4})\b")


def _mk_date(y: int, m: int, d: int) -> Optional[Date]:
    """Build a plausible Date, expanding a 2-digit year and rejecting out-of-range ones
    (so a stray ``12/34/56`` or a serial number is not read as a date)."""
    if y < 100:
        y += 2000 if y < 70 else 1900
    if not (1990 <= y <= 2100):
        return None
    try:
        return Date(y, m, d)
    except ValueError:
        return None


def _month_num(name: str) -> Optional[int]:
    return _MONTHS.get(name[:3].lower())


def _find_date(text: str) -> Optional[Date]:
    """First parseable date in ``text``. Numeric ``a/b/c`` is read month-first then
    day-first (matching the VLM extractor's format order) — genuinely ambiguous, noted."""
    m = _DATE_NUMERIC_YMD.search(text)
    if m:
        d = _mk_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return d
    m = _DATE_NUMERIC_DMY.search(text)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d = _mk_date(c, a, b) or _mk_date(c, b, a)  # MDY first, then DMY
        if d:
            return d
    m = _DATE_D_MON_Y.search(text)
    if m and _month_num(m.group(2)):
        d = _mk_date(int(m.group(3)), _month_num(m.group(2)), int(m.group(1)))
        if d:
            return d
    m = _DATE_MON_D_Y.search(text)
    if m and _month_num(m.group(1)):
        d = _mk_date(int(m.group(3)), _month_num(m.group(1)), int(m.group(2)))
        if d:
            return d
    return None


# --- money on a line ---------------------------------------------------------

def _last_money(text: str, *, require_price_shape: bool = False) -> tuple[Optional[float], Optional[str]]:
    """The right-most money value on a line, with the matched token (for stripping the
    description). When ``require_price_shape`` the token must end in a 2-decimal part,
    which screens out phone numbers / quantities when harvesting line-item prices."""
    for tok in reversed(_MONEY_TOKEN_RE.findall(text)):
        if require_price_shape and not _PRICE_SHAPE_RE.search(tok):
            continue
        val = parse_money(tok)
        if val is not None:
            return val, tok
    return None, None


# --- OCR result -> Receipt-agnostic lines ------------------------------------

@dataclass
class _Line:
    text: str
    y: float          # vertical centre in [0,1], page-top = 0
    conf: float       # mean OCR recognition confidence of the line's words, in [0,1]
    x: float = 0.0    # left edge in [0,1]; orders fragments of the same row left->right


def _geom_y_center(geom: object) -> float:
    """Mean y over a docTR geometry — ``((xmin,ymin),(xmax,ymax))`` for straight pages or
    a 4-point polygon otherwise. Defensive: an unknown shape sorts to the top (y=0)."""
    if not isinstance(geom, (list, tuple)):
        return 0.0
    ys = []
    for pt in geom:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                ys.append(float(pt[1]))
            except (TypeError, ValueError):
                pass
    return sum(ys) / len(ys) if ys else 0.0


def _geom_x_min(geom: object) -> float:
    """Left edge x over a docTR geometry (box or polygon). Used only to order the
    fragments of a merged row left-to-right. Defensive: unknown shape -> 0.0 (sorts left)."""
    if not isinstance(geom, (list, tuple)):
        return 0.0
    xs = []
    for pt in geom:
        if isinstance(pt, (list, tuple)) and len(pt) >= 1:
            try:
                xs.append(float(pt[0]))
            except (TypeError, ValueError):
                pass
    return min(xs) if xs else 0.0


def _lines_from_export(export: dict) -> list[_Line]:
    """Flatten docTR's exported result (pages -> blocks -> lines -> words) into text lines
    with a vertical position and mean recognition confidence, sorted top-to-bottom. Pure
    (takes a plain dict), so the KIE layer is unit-testable without running OCR."""
    lines: list[_Line] = []
    for page in export.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", []) or []
                text = " ".join(str(w.get("value", "")) for w in words if isinstance(w, dict)).strip()
                if not text:
                    continue
                confs = [float(w["confidence"]) for w in words
                         if isinstance(w, dict) and w.get("confidence") is not None]
                conf = sum(confs) / len(confs) if confs else 1.0
                geom = line.get("geometry")
                lines.append(_Line(text=text, y=_geom_y_center(geom), conf=conf, x=_geom_x_min(geom)))
    lines.sort(key=lambda ln: ln.y)
    return lines


def _merge_rows(lines: list[_Line], y_tol: float = 0.02) -> list[_Line]:
    """Reconstruct logical receipt *rows* from docTR's lines. docTR emits each visual
    row's left-column label and right-column amount as **separate** lines at (nearly) the
    same height, so a single-line KIE never sees a keyword and its amount together (it
    reads ``SUBTOTAL`` / ``TOTAL`` as money-less and harvests the orphaned amounts as
    fake items). We first cluster lines whose vertical centres fall within ``y_tol`` and
    join each cluster into one line, ordered left-to-right by ``x`` — so the right-column
    amount stays the **right-most** money token (``TAX1`` + ``4.48`` -> ``TAX1 4.48``, not
    ``4.48 TAX1``, which would make ``_last_money`` read the stray ``1``). y/conf/x of the
    row are the cluster means/min. Pure, so the KIE stays unit-testable without OCR.

    ``lines`` must be sorted top-to-bottom (as ``_lines_from_export`` returns them)."""
    if not lines:
        return []
    rows: list[_Line] = []
    cluster = [lines[0]]
    anchor = lines[0].y          # cluster grows while within y_tol of its first member's y
    for ln in lines[1:]:
        if abs(ln.y - anchor) <= y_tol:
            cluster.append(ln)
        else:
            rows.append(_merge_cluster(cluster))
            cluster, anchor = [ln], ln.y
    rows.append(_merge_cluster(cluster))
    return rows


def _merge_cluster(cluster: list[_Line]) -> _Line:
    """Join one same-row cluster into a single line: text x-ordered (label then amount),
    y/conf = mean, x = left-most edge."""
    ordered = sorted(cluster, key=lambda ln: ln.x)
    return _Line(
        text=" ".join(ln.text for ln in ordered),
        y=sum(ln.y for ln in cluster) / len(cluster),
        conf=sum(ln.conf for ln in cluster) / len(cluster),
        x=min(ln.x for ln in cluster),
    )


# --- KIE: lines -> Receipt ---------------------------------------------------

def _has_kw(text_low: str, kws: tuple[str, ...]) -> bool:
    return any(k in text_low for k in kws)


def _pick_vendor(lines: list[_Line], top_frac: float = 0.35) -> Optional[str]:
    """Vendor = the most letter-heavy line in the top band of the page (store names are
    the prominent text up top). Falls back to the first few lines if the band is empty."""
    top = [ln for ln in lines if ln.y <= top_frac] or lines[:3]
    best, best_alpha = None, 0
    for ln in top:
        alpha = sum(c.isalpha() for c in ln.text)
        if alpha >= 3 and alpha > best_alpha:
            best, best_alpha = ln.text, alpha
    return best


def _pick_money_field(
    lines: list[_Line], include_kw: tuple[str, ...], *, exclude_kw: tuple[str, ...] = ()
) -> Optional[tuple[float, float]]:
    """The (value, line-confidence) for the bottom-most line carrying a keyword in
    ``include_kw`` (and none in ``exclude_kw``) plus a money value. Bottom-most because
    summary fields print at the foot and we want the final figure."""
    found: Optional[tuple[float, float]] = None
    for ln in lines:  # sorted top->bottom; keep the last (lowest) match
        low = ln.text.lower()
        if _has_kw(low, include_kw) and not (exclude_kw and _has_kw(low, exclude_kw)):
            val, _ = _last_money(ln.text)
            if val is not None:
                found = (val, ln.conf)
    return found


def _pick_line_items(lines: list[_Line]) -> tuple[list[LineItem], list[float]]:
    """Non-summary lines ending in a price-shaped amount become line items, with their
    recognition confidences. Quantity defaults to 1 and unit_price to the amount, so a
    harvested item is internally consistent (the arithmetic detector never flags our own
    heuristic items; it can only flag a subtotal-vs-sum gap, which is the real signal)."""
    items: list[LineItem] = []
    confs: list[float] = []
    for ln in lines:
        if _has_kw(ln.text.lower(), _NONITEM_KW):
            continue
        val, tok = _last_money(ln.text, require_price_shape=True)
        if val is None:
            continue
        idx = ln.text.rfind(tok) if tok else -1
        desc = ln.text[:idx].strip(" \t.:-") if idx > 0 else ""
        items.append(LineItem(description=desc or "item", quantity=1.0, unit_price=val, amount=val))
        confs.append(ln.conf)
    return items, confs


def _receipt_from_lines(lines: list[_Line], doc_id: str, image_path: str) -> Receipt:
    """Map OCR'd lines onto a Receipt with the keyword/position heuristics. Pure +
    model-free (unit-tested). Records genuine OCR recognition confidence (< 1.0) under the
    keys the ``arithmetic`` guard reads, so a garbled money read makes it abstain.

    Lines are first row-merged (see :func:`_merge_rows`) so a label and its right-column
    amount, which docTR splits apart, are reunited before the single-line KIE runs."""
    rows = _merge_rows(lines)
    full_text = "\n".join(ln.text for ln in rows)
    vendor = _pick_vendor(rows)
    subtotal = _pick_money_field(rows, _SUBTOTAL_KW)
    tax = _pick_money_field(rows, _TAX_KW, exclude_kw=_SUBTOTAL_KW)
    total = _pick_money_field(rows, _TOTAL_KW, exclude_kw=_SUBTOTAL_KW)
    items, item_confs = _pick_line_items(rows)

    # Only sub-1.0 confidences are recorded, so a perfectly-read receipt leaves
    # field_confidence empty (== fully trusted) and behaviour matches the oracle path.
    field_confidence: dict[str, float] = {}
    for key, picked in (("subtotal", subtotal), ("tax_amount", tax), ("total", total)):
        if picked is not None and picked[1] < 1.0:
            field_confidence[key] = round(picked[1], 3)
    if item_confs:
        mean_item_conf = sum(item_confs) / len(item_confs)
        if mean_item_conf < 1.0:
            field_confidence["line_items"] = round(mean_item_conf, 3)

    return Receipt(
        doc_id=doc_id,
        vendor_name=vendor.strip() if vendor else "(unknown)",
        date=_find_date(full_text),
        currency="USD",
        country="US",
        line_items=items,
        subtotal=subtotal[0] if subtotal else None,
        tax_amount=tax[0] if tax else None,
        total=total[0] if total else None,
        source=DocumentType.IMAGE,
        source_path=image_path,
        image_path=image_path,
        field_confidence=field_confidence,
    )


class DocTROCRExtractor(Extractor):
    """docTR OCR (text detection + recognition) + a transparent keyword/position KIE layer
    mapping read lines onto the Receipt schema. The non-ML KIE keeps it inspectable; the
    leaderboard reports how far the heuristics get against the VLM."""

    name = "doctr"
    handles = (DocumentType.IMAGE,)

    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn") -> None:
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self._model = None

    def available(self) -> tuple[bool, str]:
        # Probe with find_spec only — never import doctr/torch here, so this stays fast
        # (the unit suite calls it) and no OCR weights are loaded.
        missing = [m for m in ("doctr", "torch", "torchvision") if not _importable(m)]
        if missing:
            return False, f"missing deps: {', '.join(missing)} — pip install python-doctr"
        return True, ""

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(
            det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True
        )

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        from doctr.io import DocumentFile

        self._ensure_model()
        doc = DocumentFile.from_images(path)
        result = self._model(doc)
        lines = _lines_from_export(result.export())
        return _receipt_from_lines(lines, doc_id=doc_id or path, image_path=path)


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None
