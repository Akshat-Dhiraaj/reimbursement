"""Confidence calibration study — does the extractor's per-value confidence actually
predict whether the value is *right*?

M2.5 wired a per-token-logprob confidence onto the scalar money fields, but the
false-positive audit showed it does not lower FP at the principled 0.5 abstain floor:
the VLM's misreads are *confident*. That is a statement about **one** threshold. This
harness asks the threshold-free question behind it — across the real corpus, is a value
the model read with **lower** confidence actually more likely to disagree with the
oracle? Three complementary views answer it:

  1. **AUC** of confidence as a predictor of correctness (``roc_auc(conf, correct)``):
     P(a correct read scored higher confidence than an incorrect one). 0.5 = the
     confidence is *uninformative* about correctness; > 0.5 = lower confidence does flag
     misreads, so a *calibrated* threshold (or a learned fuser, M3) could recover FP the
     raw 0.5 floor cannot.
  2. A **reliability table** (accuracy per confidence bin) — *where* on the scale, if
     anywhere, the reads become unreliable.
  3. A **threshold sweep** — if we abstained below T, how many misreads would we catch
     (recall) versus how many correct reads we would needlessly drop (the cost).

Ground truth is the same WildReceipt oracle the extraction benchmark trusts, on the same
image-bearing receipts the FP audit re-extracts, so these numbers line up with the ones
those harnesses already report. Honest caveat inherited from there: this is
agreement-with-the-oracle, not absolute truth — the oracle is itself imperfect, so a
"misread" is really "a read the oracle disagrees with".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..extractors.base import Extractor
from ..models import Receipt
from .extraction import _MONEY_FIELDS, _money_ok
from .metrics import roc_auc

#: reliability-table edges; the exact-1.0 mass (fields read at full confidence) gets its
#: own row because that is where most correct reads pile up and it must not be smeared
#: into a [0.9, 1.0) band.
_BIN_EDGES = (0.0, 0.6, 0.7, 0.8, 0.9, 1.0)
#: abstain thresholds swept for "drop reads with confidence < T".
_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass
class ConfRow:
    """One scored scalar read: the confidence the extractor reported for it and whether
    it matched the oracle. ``name`` is kept so we can break the study out per field."""
    name: str
    confidence: float
    correct: bool


def _confidence_of(receipt: Receipt, field_name: str) -> float:
    """The confidence the extractor attached to ``field_name``. By convention (see the
    VLM extractor) only *sub-1.0* confidences are recorded, so a field absent from the
    map was read at full confidence -> 1.0."""
    return receipt.field_confidence.get(field_name, 1.0)


def collect_confidence_rows(
    extractor: Extractor, truths: Sequence[Receipt]
) -> list[ConfRow]:
    """Run ``extractor`` on each oracle receipt's image and, for every money field the
    oracle can score, record (confidence, correct-vs-oracle). Mirrors
    ``evaluate_extractor``'s loop but keeps the per-read confidence instead of only a
    hit/miss tally. A field is recorded only when *both* the oracle has a value (so
    correctness is decidable) and the extractor returned one (so there is a read to
    score): a field the model omitted has no token to be confident about, and belongs to
    the parse-completeness signal, not this one."""
    rows: list[ConfRow] = []
    for truth in truths:
        try:
            pred: Optional[Receipt] = extractor.extract(
                truth.image_path or "", doc_id=truth.doc_id
            )
        except Exception:  # an unreadable doc has no confidence rows, not a crash
            pred = None
        if pred is None:
            continue
        for name in _MONEY_FIELDS:
            tv, pv = getattr(truth, name), getattr(pred, name)
            if tv is None or pv is None:
                continue
            rows.append(ConfRow(name, _confidence_of(pred, name), _money_ok(pv, tv)))
    return rows


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class FieldCalibration:
    """Separation of correct from incorrect reads for one field (or the pooled set)."""
    name: str
    n: int
    n_incorrect: int
    auc: float                 # P(conf(correct) > conf(incorrect)); 0.5 == uninformative
    mean_conf_correct: float
    mean_conf_incorrect: float

    @property
    def accuracy(self) -> float:
        return (self.n - self.n_incorrect) / self.n if self.n else float("nan")


def _calibrate(name: str, rows: Sequence[ConfRow]) -> FieldCalibration:
    return FieldCalibration(
        name=name,
        n=len(rows),
        n_incorrect=sum(not r.correct for r in rows),
        auc=roc_auc([r.confidence for r in rows], [r.correct for r in rows]),
        mean_conf_correct=_mean([r.confidence for r in rows if r.correct]),
        mean_conf_incorrect=_mean([r.confidence for r in rows if not r.correct]),
    )


@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")


def _reliability(rows: Sequence[ConfRow]) -> list[Bin]:
    """Accuracy within each confidence band, plus a dedicated exact-1.0 row."""
    bins: list[Bin] = []
    for lo, hi in zip(_BIN_EDGES, _BIN_EDGES[1:]):
        sel = [r for r in rows if lo <= r.confidence < hi]
        bins.append(Bin(lo, hi, len(sel), sum(r.correct for r in sel)))
    full = [r for r in rows if r.confidence >= 1.0]
    bins.append(Bin(1.0, 1.0, len(full), sum(r.correct for r in full)))
    return bins


@dataclass
class Sweep:
    threshold: float
    misreads_caught: int   # incorrect reads with conf < T (the win)
    total_incorrect: int
    correct_abstained: int  # correct reads with conf < T (the cost)
    total_correct: int

    @property
    def recall(self) -> float:
        return self.misreads_caught / self.total_incorrect if self.total_incorrect else float("nan")

    @property
    def cost(self) -> float:
        return self.correct_abstained / self.total_correct if self.total_correct else float("nan")


def _sweep(rows: Sequence[ConfRow]) -> list[Sweep]:
    inc = [r for r in rows if not r.correct]
    cor = [r for r in rows if r.correct]
    return [
        Sweep(t,
              sum(r.confidence < t for r in inc), len(inc),
              sum(r.confidence < t for r in cor), len(cor))
        for t in _THRESHOLDS
    ]


@dataclass
class CalibrationReport:
    name: str                       # the extractor whose confidence we calibrated
    per_field: list[FieldCalibration] = field(default_factory=list)
    bins: list[Bin] = field(default_factory=list)
    sweep: list[Sweep] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Confidence calibration: {self.name} — does lower confidence flag a misread?",
            "(AUC = P[correct read scored higher than an incorrect one]; 0.5 = uninformative)",
            "",
            f"{'field':12} {'n':>5} {'wrong':>6} {'acc':>6} {'AUC':>6} "
            f"{'conf|ok':>8} {'conf|wrong':>11}",
            "-" * 62,
        ]
        for f in self.per_field:
            lines.append(
                f"{f.name:12} {f.n:>5} {f.n_incorrect:>6} {_fmt(f.accuracy):>6} "
                f"{_fmt(f.auc):>6} {_fmt(f.mean_conf_correct):>8} "
                f"{_fmt(f.mean_conf_incorrect):>11}"
            )
        lines += [
            "",
            "Reliability (is a low-confidence band actually less accurate?):",
            f"  {'confidence':>12} {'n':>5} {'accuracy':>9}",
        ]
        for b in self.bins:
            label = "==1.0 (full)" if b.lo == 1.0 else f"[{b.lo:.1f}, {b.hi:.1f})"
            lines.append(f"  {label:>12} {b.n:>5} {_fmt(b.accuracy):>9}")
        lines += [
            "",
            "Abstain sweep (drop reads with confidence < T):",
            f"  {'T':>4} {'misreads_caught':>16} {'correct_dropped':>16}",
        ]
        for s in self.sweep:
            lines.append(
                f"  {s.threshold:>4.2f} "
                f"{f'{s.misreads_caught}/{s.total_incorrect} ({_fmt(s.recall)})':>16} "
                f"{f'{s.correct_abstained}/{s.total_correct} ({_fmt(s.cost)})':>16}"
            )
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:.3f}"


def summarize_calibration(
    rows: Sequence[ConfRow], extractor_name: str = "extractor"
) -> CalibrationReport:
    """Turn the collected per-read rows into the three calibration views. Pure and
    GPU-free, so the (slow) collection can run once and the analysis stay unit-testable
    and cheap to re-read."""
    rows = list(rows)
    per_field = [_calibrate("ALL", rows)]
    for name in _MONEY_FIELDS:
        sub = [r for r in rows if r.name == name]
        if sub:
            per_field.append(_calibrate(name, sub))
    return CalibrationReport(
        name=extractor_name,
        per_field=per_field,
        bins=_reliability(rows),
        sweep=_sweep(rows),
    )
