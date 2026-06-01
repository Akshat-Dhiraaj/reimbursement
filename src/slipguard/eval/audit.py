"""False-positive audit on a corpus assumed entirely legitimate.

The synthetic benchmark measures recall on fraud we minted ourselves; it cannot
measure how often the detectors cry wolf on *genuine* receipts. This audit does
exactly that: every receipt here is real and legitimate, so **any** flag
(review/reject) is a false positive. It reports the fused FP rate, a per-detector
breakdown (abstain / active / own-flag rate), field coverage of the extractor
that produced the receipts, and a categorised view of *why* the arithmetic
detector fired — because on real data the binding constraint is usually
extraction completeness, not detector logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..detectors.base import Detector
from ..fusion import Fuser
from ..models import Decision, Receipt

if TYPE_CHECKING:  # only for the type hint — avoid importing the extractors package here
    from ..extractors.base import Extractor

# arithmetic reason text -> bucket, so we can separate "extraction was lossy"
# (subtotal/total disagree because not every line was captured) from a genuine
# field-level contradiction.
_ARITH_BUCKETS = (
    ("sum(line items)", "subtotal!=sum(lines)"),
    ("subtotal+tax", "total!=subtotal+tax"),
    ("qty*price", "line!=qty*price"),
)


@dataclass
class DetectorFP:
    name: str
    n_abstain: int
    n_active: int
    n_flag: int  # active receipts whose effective_score >= flag_threshold
    sum_active_score: float

    @property
    def abstain_rate(self) -> float:
        total = self.n_abstain + self.n_active
        return self.n_abstain / total if total else 0.0

    @property
    def flag_rate(self) -> float:
        total = self.n_abstain + self.n_active
        return self.n_flag / total if total else 0.0

    @property
    def mean_active(self) -> float:
        return self.sum_active_score / self.n_active if self.n_active else 0.0


@dataclass
class Example:
    doc_id: str
    vendor: str
    decision: str
    risk: float
    reasons: list[str]


@dataclass
class FalsePositiveAudit:
    n: int
    decisions: dict[str, int]
    detectors: list[DetectorFP]
    field_coverage: dict[str, int]
    arithmetic_reasons: dict[str, int]
    examples: list[Example]
    flag_threshold: float
    review_threshold: float

    @property
    def n_flagged(self) -> int:
        return self.decisions.get("review", 0) + self.decisions.get("reject", 0)

    @property
    def fp_rate(self) -> float:
        return self.n_flagged / self.n if self.n else 0.0

    def __str__(self) -> str:
        cov = self.field_coverage
        lines = [
            f"False-positive audit: {self.n} legitimate receipts "
            f"(any flag = false positive)",
            "",
            f"Fused FP rate: {self.fp_rate:.3f}  ({self.n_flagged}/{self.n} flagged)",
            "Decisions: " + ", ".join(f"{k}={v}" for k, v in self.decisions.items()),
            "",
            f"{'detector':16} {'abstain%':>9} {'flag%':>7} {'active_mean':>12}",
            "-" * 48,
        ]
        for d in self.detectors:
            lines.append(
                f"{d.name:16} {100 * d.abstain_rate:8.1f}% {100 * d.flag_rate:6.1f}% "
                f"{d.mean_active:12.3f}"
            )
        lines += [
            "-" * 48,
            "",
            "Extractor field coverage (how often each field was present):",
            "  " + ", ".join(f"{k}={v}/{self.n}" for k, v in cov.items()),
        ]
        if self.arithmetic_reasons:
            lines += [
                "",
                "Arithmetic failures by cause "
                "(lossy extraction vs. genuine contradiction):",
                "  " + ", ".join(f"{k}={v}" for k, v in self.arithmetic_reasons.items()),
            ]
        if self.examples:
            lines += ["", f"Example flagged receipts (first {len(self.examples)}):"]
            for e in self.examples:
                lines.append(
                    f"  {e.doc_id} [{e.vendor[:24]}] {e.decision.upper()} "
                    f"risk={e.risk:.3f}"
                )
                for r in e.reasons:
                    lines.append(f"      - {r}")
        return "\n".join(lines)


def image_bearing(
    receipts: Sequence[Receipt], limit: Optional[int] = None
) -> list[Receipt]:
    """The receipts an image-route extractor can open (those with an ``image_path``),
    optionally capped at the first ``limit``. This is the exact subset ``reextract``
    re-extracts, so auditing the oracle on ``image_bearing(receipts, N)`` compares it
    against a VLM/OCR re-extraction on the **identical** N receipts — apples-to-apples,
    not 472-oracle vs. 100-re-extracted."""
    sourced = [r for r in receipts if r.image_path]
    return sourced[:limit] if limit else sourced


def reextract(
    extractor: "Extractor",
    receipts: Sequence[Receipt],
    limit: Optional[int] = None,
) -> list[Receipt]:
    """Re-extract each receipt straight from its source image via a real ``Extractor``,
    so the FP audit runs on *faithfully-extracted* fields — with the extractor's
    ``field_confidence`` live — instead of the oracle's lossy KIE reconstruction. This
    is what lets us measure arithmetic's TRUE false-positive rate (the audit named
    extraction quality, not detector logic, as the binding constraint). Receipts with no
    ``image_path`` are skipped; ``limit`` caps the count for a slow VLM on a laptop."""
    sourced = image_bearing(receipts, limit)
    return [extractor.extract(r.image_path, doc_id=r.doc_id) for r in sourced]


def _coverage(receipts: Sequence[Receipt]) -> dict[str, int]:
    return {
        "date": sum(r.date is not None for r in receipts),
        "line_items": sum(bool(r.line_items) for r in receipts),
        "subtotal": sum(r.subtotal is not None for r in receipts),
        "tax_amount": sum(r.tax_amount is not None for r in receipts),
        "total": sum(r.total is not None for r in receipts),
    }


def audit_false_positives(
    receipts: Sequence[Receipt],
    detectors: Sequence[Detector],
    fuser: Optional[Fuser] = None,
    flag_threshold: float = 0.5,
    max_examples: int = 12,
) -> FalsePositiveAudit:
    """Run ``detectors`` + ``fuser`` over a legitimate corpus and tally false
    positives. Detectors are primed with empty history: each receipt is judged on
    its own merits (priming with the corpus itself would manufacture duplicate
    matches between distinct legitimate receipts)."""
    fuser = fuser or Fuser()
    for d in detectors:
        d.prime([])

    stats = {d.name: DetectorFP(d.name, 0, 0, 0, 0.0) for d in detectors}
    arith_reasons: dict[str, int] = {}
    decisions = {dec.value: 0 for dec in Decision}
    examples: list[Example] = []

    for r in receipts:
        signals = [d.run(r) for d in detectors]
        for sig in signals:
            st = stats[sig.detector]
            if sig.abstained:
                st.n_abstain += 1
                continue
            st.n_active += 1
            st.sum_active_score += sig.score
            if sig.score >= flag_threshold:
                st.n_flag += 1
            if sig.detector == "arithmetic":
                for text, bucket in _ARITH_BUCKETS:
                    if any(text in reason for reason in sig.reasons):
                        arith_reasons[bucket] = arith_reasons.get(bucket, 0) + 1

        verdict = fuser.verdict(r.doc_id, signals)
        decisions[verdict.decision.value] += 1
        if verdict.decision is not Decision.APPROVE and len(examples) < max_examples:
            examples.append(
                Example(r.doc_id, r.vendor_name, verdict.decision.value,
                        verdict.risk_score, verdict.reasons[:3])
            )

    return FalsePositiveAudit(
        n=len(receipts),
        decisions=decisions,
        detectors=list(stats.values()),
        field_coverage=_coverage(receipts),
        arithmetic_reasons=arith_reasons,
        examples=examples,
        flag_threshold=flag_threshold,
        review_threshold=fuser.review_threshold,
    )
