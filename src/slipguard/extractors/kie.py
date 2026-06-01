"""Keyword/position KIE — map positioned text lines onto a Receipt.

This is the paradigm-agnostic half of an extractor: given a document's text as
*positioned lines* (text + vertical position + a per-line confidence + left edge),
it picks vendor / date / subtotal / tax / total / line-items with small, transparent
keyword-and-position heuristics. It knows nothing about *how* the lines were read.

Two extractors share it, which is exactly why it lives here rather than inside one of
them:

* :mod:`slipguard.extractors.doctr_ocr` — lines come from an OCR recogniser (text
  detection + recognition); the per-line confidence is a genuine recognition softmax.
* :mod:`slipguard.extractors.pdf_text` — lines come from a born-digital PDF's embedded
  text layer (pypdfium2); the text is exact, so confidence is a constant 1.0.

The KIE is pure and model-free, so it is unit-tested directly without loading any OCR
weights or opening any PDF. ``Line`` is the contract between an extractor's reader and
this layer.

Honest limit: every confidence flowing through here is a *reading* confidence (did we
read these glyphs / this text run correctly), not a field-*labelling* confidence (did we
pick the right line as "total"). It catches a garbled or low-quality read, not a
confident misread of the wrong row — there is no honest signal for the latter without
ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

from ..models import DocumentType, LineItem, Receipt

from ..money import parse_money

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


# --- positioned text line: the contract between a reader and this KIE --------

@dataclass
class Line:
    """One positioned line of text from whatever read the document.

    ``y`` is the vertical centre in [0,1] with page-top = 0 (so "top band" and
    "bottom-most" are comparisons on ``y``); ``conf`` is the reader's per-line
    confidence in [0,1] (OCR recognition softmax, or 1.0 for exact digital text); ``x``
    is the left edge in [0,1], used only to order fragments of the same row left->right.
    """

    text: str
    y: float
    conf: float
    x: float = 0.0


def _merge_rows(lines: list[Line], y_tol: float = 0.02) -> list[Line]:
    """Reconstruct logical receipt *rows* from a reader's lines. Some readers (docTR;
    some PDF layouts) emit a row's left-column label and right-column amount as
    **separate** lines at (nearly) the same height, so a single-line KIE never sees a
    keyword and its amount together (it reads ``SUBTOTAL`` / ``TOTAL`` as money-less and
    harvests the orphaned amounts as fake items). We cluster lines whose vertical centres
    fall within ``y_tol`` and join each cluster into one line, ordered left-to-right by
    ``x`` — so the right-column amount stays the **right-most** money token (``TAX1`` +
    ``4.48`` -> ``TAX1 4.48``, not ``4.48 TAX1``, which would make ``_last_money`` read
    the stray ``1``). y/conf/x of the row are the cluster means/min. Pure, so the KIE
    stays unit-testable without a reader.

    ``lines`` must be sorted top-to-bottom."""
    if not lines:
        return []
    rows: list[Line] = []
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


def _merge_cluster(cluster: list[Line]) -> Line:
    """Join one same-row cluster into a single line: text x-ordered (label then amount),
    y/conf = mean, x = left-most edge."""
    ordered = sorted(cluster, key=lambda ln: ln.x)
    return Line(
        text=" ".join(ln.text for ln in ordered),
        y=sum(ln.y for ln in cluster) / len(cluster),
        conf=sum(ln.conf for ln in cluster) / len(cluster),
        x=min(ln.x for ln in cluster),
    )


# --- KIE: lines -> Receipt ---------------------------------------------------

def _has_kw(text_low: str, kws: tuple[str, ...]) -> bool:
    return any(k in text_low for k in kws)


def _pick_vendor(lines: list[Line], top_frac: float = 0.35) -> Optional[str]:
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
    lines: list[Line], include_kw: tuple[str, ...], *, exclude_kw: tuple[str, ...] = ()
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


def _pick_line_items(lines: list[Line]) -> tuple[list[LineItem], list[float]]:
    """Non-summary lines ending in a price-shaped amount become line items, with their
    reading confidences. Quantity defaults to 1 and unit_price to the amount, so a
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


def receipt_from_lines(
    lines: list[Line],
    doc_id: str,
    *,
    source: DocumentType,
    source_path: str,
    image_path: Optional[str] = None,
) -> Receipt:
    """Map positioned text lines onto a Receipt with the keyword/position heuristics. Pure
    and model-free (unit-tested). Records the reader's per-line confidence (< 1.0) under
    the field keys the ``arithmetic`` guard reads, so a low-quality read makes it abstain.

    Lines are first row-merged (see :func:`_merge_rows`) so a label and its right-column
    amount, which some readers split apart, are reunited before the single-line KIE runs.
    ``source`` / ``source_path`` set the provenance route (PDF vs IMAGE) so the right
    forensic detector runs; ``image_path`` is set only on the IMAGE route."""
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
        source=source,
        source_path=source_path,
        image_path=image_path,
        field_confidence=field_confidence,
    )
