"""Run each detector independently on a labelled dataset, then the fused verdict,
and report ranked metrics. Selection is driven by these numbers, not by priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..data.synth import Dataset
from ..detectors.base import Detector
from ..fusion import Fuser
from ..models import Decision, FraudType
from .metrics import false_positive_rate, precision_recall_f1, roc_auc


@dataclass
class DetectorReport:
    name: str
    targets: list[str]
    auc: float          # detecting *any* fraud (single-purpose detectors score low here by design)
    target_recall: float  # recall on its own fraud subtype
    fp_rate: float        # false positives on clean receipts
    n_target: int


@dataclass
class FusionReport:
    auc: float
    precision: float
    recall: float
    f1: float
    fp_rate: float
    decisions: dict[str, int] = field(default_factory=dict)
    subtype_recall: dict[str, float] = field(default_factory=dict)


@dataclass
class Report:
    detectors: list[DetectorReport]
    fusion: FusionReport
    n_samples: int
    n_fraud: int

    def __str__(self) -> str:  # human-readable leaderboard
        pct = 100 * self.n_fraud / self.n_samples if self.n_samples else 0
        lines = [
            f"Benchmark: {self.n_samples} samples, {self.n_fraud} fraud ({pct:.0f}%)",
            "",
            f"{'detector':16} {'targets':14} {'AUC':>6} {'t-recall':>9} {'FP':>6} {'n_tgt':>6}",
            "-" * 62,
        ]
        for d in self.detectors:
            lines.append(
                f"{d.name:16} {','.join(d.targets)[:14]:14} {_fmt(d.auc):>6} "
                f"{_fmt(d.target_recall):>9} {_fmt(d.fp_rate):>6} {d.n_target:>6}"
            )
        lines.append("-" * 62)
        f = self.fusion
        lines.append(
            f"{'FUSED':16} {'any':14} {_fmt(f.auc):>6} {_fmt(f.recall):>9} "
            f"{_fmt(f.fp_rate):>6} {self.n_fraud:>6}"
        )
        lines += [
            "",
            f"Fused: precision={_fmt(f.precision)} recall={_fmt(f.recall)} "
            f"f1={_fmt(f.f1)} fp={_fmt(f.fp_rate)}",
            "Decisions: " + ", ".join(f"{k}={v}" for k, v in f.decisions.items()),
            "Per-subtype recall (fused): "
            + ", ".join(f"{k}={_fmt(v)}" for k, v in f.subtype_recall.items()),
        ]
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:5.3f}"


def evaluate(
    dataset: Dataset,
    detectors: Sequence[Detector],
    fuser: Optional[Fuser] = None,
    flag_threshold: float = 0.5,
) -> Report:
    fuser = fuser or Fuser()
    samples = dataset.samples
    labels = [s.is_fraud for s in samples]
    clean_idx = [i for i, s in enumerate(samples) if not s.is_fraud]

    for d in detectors:
        d.prime(dataset.history)

    # signals[i] = list of Signal across detectors, reused for fusion
    signals: list[list] = [[] for _ in samples]
    det_reports: list[DetectorReport] = []

    for d in detectors:
        scores: list[float] = []
        for i, s in enumerate(samples):
            sig = d.run(s.receipt)
            signals[i].append(sig)
            scores.append(sig.effective_score)

        auc = roc_auc(scores, labels)
        tgt_idx = [i for i, s in enumerate(samples) if s.fraud_types & d.targets]
        t_recall = (
            sum(scores[i] >= flag_threshold for i in tgt_idx) / len(tgt_idx)
            if tgt_idx else float("nan")
        )
        fp = (
            sum(scores[i] >= flag_threshold for i in clean_idx) / len(clean_idx)
            if clean_idx else float("nan")
        )
        det_reports.append(
            DetectorReport(d.name, [t.value for t in d.targets], auc, t_recall, fp, len(tgt_idx))
        )

    # fusion
    risks: list[float] = []
    decisions: list[Decision] = []
    for i, s in enumerate(samples):
        v = fuser.verdict(s.receipt.doc_id, signals[i])
        risks.append(v.risk_score)
        decisions.append(v.decision)

    flagged = [r >= fuser.review_threshold for r in risks]  # review or reject
    prec, rec, f1 = precision_recall_f1(flagged, labels)

    subtype_recall: dict[str, float] = {}
    for ft in FraudType:
        if ft is FraudType.NONE:
            continue
        idx = [i for i, s in enumerate(samples) if ft in s.fraud_types]
        if idx:
            subtype_recall[ft.value] = sum(flagged[i] for i in idx) / len(idx)

    fusion = FusionReport(
        auc=roc_auc(risks, labels),
        precision=prec, recall=rec, f1=f1,
        fp_rate=false_positive_rate(flagged, labels),
        decisions={d.value: sum(x == d for x in decisions) for d in Decision},
        subtype_recall=subtype_recall,
    )
    return Report(det_reports, fusion, len(samples), sum(labels))
