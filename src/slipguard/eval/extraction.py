"""Extraction-accuracy benchmark — rank extractors by how faithfully they read a
document into a Receipt, exactly as ``harness.py`` ranks detectors by fraud-catching.

The real-data audit proved arithmetic precision is capped by extraction quality, so
*which* extractor we ship has to be picked on measured field accuracy, not reputation.
This harness supplies that number.

Ground truth is a corpus of Receipts whose fields are already trusted — WildReceipt's
human KIE labels give an *oracle* Receipt per image (see ``data/wildreceipt.py``). A
candidate extractor is run on the same image and its output is compared field-by-field
against the oracle. The oracle is the reference; the OCR/VLM extractor is the
prediction — so the comparison is honest and non-circular.

We only score a field when the oracle actually has a value for it, so an extractor is
never penalised for a field ground truth itself is missing. A field present in truth
but absent/wrong in the prediction (including when extraction raises) counts as a miss.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..detectors.duplicate import _norm_vendor  # alnum-lowercase normaliser (DRY)
from ..extractors.base import Extractor
from ..models import Receipt
from ..money import money_close
from .metrics import _fmt

#: Receipt money fields scored numerically with a tolerance.
_MONEY_FIELDS = ("subtotal", "tax_amount", "total")
#: vendor placeholders the oracle emits when it has no name — don't score those.
_UNKNOWN_VENDORS = {"", "(unknown)"}
#: fixed field order for the report table.
_FIELDS = ("vendor", "date", *_MONEY_FIELDS, "line_count")


def _vendor_ok(pred: Optional[str], truth: str, ratio: float = 0.8) -> bool:
    """Vendor match on the normalised strings (same normaliser the duplicate detector
    uses). A blank prediction is always a miss.

    Containment counts as a match in *either* direction: WildReceipt's ``Store_name``
    label is often a terse token (``COSTCO``) or carries OCR junk (``RoyalHotel
    TaxInvoice``), while a good extractor returns the fuller real name (``Costco
    Wholesale``). Penalising that as wrong measures agreement-with-the-label, not
    extraction quality — so if one normalised name contains the other (with a length
    floor so a 1-2 char fragment can't match everything) we credit it; otherwise we
    fall back to a fuzzy ratio for near-misreads (``Trader Joe's`` vs ``Trader Joes``)."""
    a, b = _norm_vendor(pred or ""), _norm_vendor(truth or "")
    if not a or not b:
        return False
    if min(len(a), len(b)) >= 4 and (a in b or b in a):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def _money_ok(pred: Optional[float], truth: float, rel: float = 0.01, abs_tol: float = 0.02) -> bool:
    """Money match within the larger of an absolute and a relative tolerance (shared
    ``money_close``) — mirrors the slack the ``arithmetic`` detector allows for rounding.
    A missing prediction (``None``) is always a miss."""
    if pred is None:
        return False
    return money_close(pred, truth, rel=rel, abs_tol=abs_tol)


@dataclass
class FieldAccuracy:
    field: str
    n: int = 0        # times the oracle had this field (so it was scorable)
    correct: int = 0  # times the prediction matched

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else float("nan")


@dataclass
class ExtractionReport:
    name: str
    n_samples: int
    n_errors: int                       # documents where extract() raised
    fields: list[FieldAccuracy] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Macro-average accuracy over scored fields (each field weighted equally,
        so a rare field can't be drowned out by a common one)."""
        accs = [f.accuracy for f in self.fields if f.n]
        return sum(accs) / len(accs) if accs else float("nan")

    def __str__(self) -> str:
        lines = [
            f"Extraction: {self.name} — {self.n_samples} receipts, "
            f"{self.n_errors} extractor error(s)",
            "",
            f"{'field':14} {'n':>6} {'correct':>8} {'accuracy':>9}",
            "-" * 40,
        ]
        for f in self.fields:
            lines.append(f"{f.field:14} {f.n:>6} {f.correct:>8} {_fmt(f.accuracy):>9}")
        lines += ["-" * 40, f"{'OVERALL (macro)':30} {_fmt(self.overall):>9}"]
        return "\n".join(lines)


def evaluate_extractor(extractor: Extractor, truths: Sequence[Receipt]) -> ExtractionReport:
    """Run ``extractor`` on every truth's source image and score its output against
    the oracle fields. ``truths`` are the oracle Receipts (``image_path`` resolved)."""
    counts = {f: FieldAccuracy(f) for f in _FIELDS}
    n_errors = 0

    for truth in truths:
        # IMAGE-route oracles carry image_path; PDF-route oracles carry source_path —
        # resolve either so the same harness scores both routes.
        doc_path = truth.image_path or truth.source_path or ""
        try:
            pred: Optional[Receipt] = extractor.extract(doc_path, doc_id=truth.doc_id)
        except Exception:  # an extractor that can't read this doc is a miss, not a crash
            pred, n_errors = None, n_errors + 1

        # Each field is scored only when the oracle supplies a value to compare to.
        if truth.vendor_name not in _UNKNOWN_VENDORS:
            fa = counts["vendor"]
            fa.n += 1
            fa.correct += _vendor_ok(pred.vendor_name if pred else None, truth.vendor_name)

        if truth.date is not None:
            fa = counts["date"]
            fa.n += 1
            fa.correct += bool(pred and pred.date == truth.date)

        for name in _MONEY_FIELDS:
            tv = getattr(truth, name)
            if tv is not None:
                fa = counts[name]
                fa.n += 1
                fa.correct += _money_ok(getattr(pred, name) if pred else None, tv)

        if truth.line_items:
            fa = counts["line_count"]
            fa.n += 1
            fa.correct += bool(pred and len(pred.line_items) == len(truth.line_items))

    return ExtractionReport(extractor.name, len(truths), n_errors, [counts[f] for f in _FIELDS])


def evaluate_extractors(
    extractors: Iterable[Extractor], truths: Sequence[Receipt]
) -> list[ExtractionReport]:
    """Rank a set of extractors on the same ground truth — the extractor leaderboard."""
    truths = list(truths)
    return [evaluate_extractor(ex, truths) for ex in extractors]
