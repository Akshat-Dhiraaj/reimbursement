"""Measure whether the learned logistic fuser beats noisy-OR — by numbers, with no
leakage and an honest operating-point comparison.

The question is the audit's: *at a matched fraud-recall, does the learned fuser flag
fewer genuine receipts?* So we treat SYNTHETIC fraud as the positives and REAL
legitimate receipts (WildReceipt / CORD) as the negatives — the false-positive
population we actually care about — plus synthetic clean receipts. The model is fit
on one synthetic seed + one half of the real corpora, and every number below is read
on a *different* synthetic seed + the held-out half (no receipt is both trained and
scored).

Caveat repeated from the model: positives are synthetic, so this is *synthetic-fraud
vs real-legitimate* separation, not real-fraud detection. The richer-``Receipt``-model
work remains the complementary, bigger lever on real FP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..data.synth import Dataset
from ..detectors import default_detectors
from ..detectors.base import Detector
from ..fusion import Fuser
from ..fusion_learned import LearnedFuser, feature_names, feature_vector
from ..models import Receipt, Signal
from .metrics import _fmt, roc_auc

Row = tuple[list[Signal], bool]


def _signals_for(
    receipts: Sequence[Receipt], detectors: Sequence[Detector], history: Sequence[Receipt]
) -> list[list[Signal]]:
    """Prime then run every detector on every receipt. Detectors are stateful via
    ``prime`` (duplicate detection), so callers re-prime per batch; this materialises
    each batch fully before the next re-prime."""
    for d in detectors:
        d.prime(history)
    return [[d.run(r) for d in detectors] for r in receipts]


def _real_detectors(real: Sequence[Receipt]) -> list[Detector]:
    """Canonical detectors **minus ``date_sanity``** for the real-corpus fusion audit.

    Real corpora (CORD / ExpressExpense, 2018-2019) predate the 60-day reimbursement window
    by years, so ``date_sanity`` would — correctly, per policy — flag *every* receipt as
    out-of-window. That is a policy signal, not a fraud-detector false positive, and it would
    swamp the fused FP / AUC metric. Dropping it isolates what the content / structure detectors
    (arithmetic, tax_id, duplicate, provenance) contribute on aged real data. In production the
    60-day window is still enforced deterministically by ``date_sanity``.

    (Before the window was tightened to 60 days, this instead pinned ``date_sanity`` to
    ``max(date) + 1yr`` to keep it quiet — under a 2-month window no single ``today`` can silence
    it across a multi-year corpus, so we simply exclude it here.)"""
    return [d for d in default_detectors() if d.name != "date_sanity"]


def _risks(rows: Sequence[list[Signal]], fuser: Fuser) -> list[float]:
    return [fuser.risk(sigs) for sigs in rows]


def _threshold_for_recall(pos_scores: Sequence[float], recall: float) -> float:
    """Highest threshold whose recall over the positives is >= ``recall`` (the
    tightest cut for that sensitivity, i.e. the fewest false positives)."""
    s = sorted(pos_scores, reverse=True)
    if not s:
        return float("inf")  # no positives -> unreachable threshold -> 0 false positives
    k = max(1, min(len(s), math.ceil(recall * len(s))))
    return s[k - 1]


def _rate_at_least(scores: Sequence[float], threshold: float) -> float:
    return sum(x >= threshold for x in scores) / len(scores) if scores else float("nan")


@dataclass
class OperatingPoint:
    """At a target synthetic fraud-recall, each fuser's own threshold to hit it and
    the resulting false-positive rate on the held-out real receipts."""

    target_recall: float
    noisy_fp: float
    learned_fp: float


@dataclass
class FusionComparison:
    corpora: list[str]
    n_train_pos: int
    n_train_neg_synth: int
    n_train_neg_real: int
    n_test_fraud: int
    n_test_clean: int
    n_test_real: int
    synth_auc_noisy: float   # synthetic fraud vs synthetic clean (the easy, internal case)
    synth_auc_learned: float
    real_auc_noisy: float    # synthetic fraud vs REAL legitimate (the separation we care about)
    real_auc_learned: float
    points: list[OperatingPoint]
    weights: dict[str, float] = field(default_factory=dict)
    #: detectors that never fired in training (all-zero feature column) — their weight
    #: is meaningless, so we flag rather than silently report a learned 0.
    inactive: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        corp = ", ".join(self.corpora) if self.corpora else "(none present)"
        lines = [
            "Learned fusion vs noisy-OR  (logistic over the same per-detector signals)",
            f"  train: {self.n_train_pos} synth-fraud + "
            f"{self.n_train_neg_synth} synth-clean + {self.n_train_neg_real} real-legit",
            f"  test : {self.n_test_fraud} synth-fraud + "
            f"{self.n_test_clean} synth-clean + {self.n_test_real} real-legit  "
            f"(disjoint from train)",
            f"  real corpora (negatives): {corp}",
            "  NOTE: positives are SYNTHETIC fraud -> this is synth-fraud vs real-legit",
            "        separation, not real-fraud detection.",
            "",
            f"{'separation (AUC)':30} {'noisy-OR':>9} {'learned':>9}",
            "-" * 50,
            f"{'synthetic fraud vs synth-clean':30} {_fmt(self.synth_auc_noisy):>9} "
            f"{_fmt(self.synth_auc_learned):>9}",
            f"{'synthetic fraud vs REAL-legit':30} {_fmt(self.real_auc_noisy):>9} "
            f"{_fmt(self.real_auc_learned):>9}",
            "",
            "False-positive rate on real receipts at a matched synthetic fraud-recall:",
            f"  {'target recall':>14} {'noisy-OR FP':>12} {'learned FP':>12}",
            "  " + "-" * 40,
        ]
        for p in self.points:
            lines.append(
                f"  {p.target_recall:>14.2f} {_fmt(p.noisy_fp):>12} {_fmt(p.learned_fp):>12}"
            )
        if self.weights:
            lines += [
                "",
                "Learned per-detector weights (legible fusion rule; |w| desc):",
                "  " + ", ".join(f"{k}={v:+.3f}" for k, v in self.weights.items()),
            ]
        if self.inactive:
            lines += [
                f"  (never fired in training, so weight is uninformative: "
                f"{', '.join(self.inactive)} — this training data is structured/KIE only, "
                f"with no PDF/image provenance examples to calibrate those detectors)",
            ]
        return "\n".join(lines)


def compare_fusion(
    train: Dataset,
    test: Dataset,
    real_receipts: Sequence[Receipt],
    *,
    corpora: Optional[Sequence[str]] = None,
    target_recalls: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
    **fit_kw,
) -> FusionComparison:
    """Fit a :class:`LearnedFuser` on ``train`` (synthetic) + the first half of
    ``real_receipts``, then compare it against noisy-OR on ``test`` (a different
    synthetic seed) + the held-out second half of the real receipts.

    Real receipts are all legitimate, so any flag on them is a false positive; they
    are split by index so the train/test halves are disjoint and reproducible."""
    real = list(real_receipts)
    half = len(real) // 2
    real_train, real_test = real[:half], real[half:]
    order = feature_names()

    # --- TRAIN rows: synthetic (fraud + clean) primed with its own history, plus
    #     real-legit primed empty (each judged on its own merits, like the audit). ---
    train_synth = _signals_for([s.receipt for s in train.samples], default_detectors(), train.history)
    train_rows: list[Row] = list(zip(train_synth, [s.is_fraud for s in train.samples]))
    n_pos = sum(1 for _, lab in train_rows if lab)
    n_neg_synth = len(train_rows) - n_pos
    if real_train:
        for sigs in _signals_for(real_train, _real_detectors(real_train), []):
            train_rows.append((sigs, False))

    learned = LearnedFuser.fit(train_rows, order, **fit_kw)
    noisy_fuser = Fuser()
    learned_fuser = Fuser(combiner=learned)

    # Detectors whose feature was 0 across every training row never had a chance to
    # earn a meaningful weight (they always abstained on this data) — flag them so a
    # silent learned 0 isn't read as "the model judged this detector useless".
    X_train = [feature_vector(sigs, order) for sigs, _ in train_rows]
    inactive = [order[j] for j in range(len(order)) if all(row[j] == 0.0 for row in X_train)]

    # --- TEST rows ---
    test_synth = _signals_for([s.receipt for s in test.samples], default_detectors(), test.history)
    fraud_rows = [sig for sig, s in zip(test_synth, test.samples) if s.is_fraud]
    clean_rows = [sig for sig, s in zip(test_synth, test.samples) if not s.is_fraud]
    real_rows = _signals_for(real_test, _real_detectors(real_test), []) if real_test else []

    f_noisy, f_learn = _risks(fraud_rows, noisy_fuser), _risks(fraud_rows, learned_fuser)
    c_noisy, c_learn = _risks(clean_rows, noisy_fuser), _risks(clean_rows, learned_fuser)
    r_noisy, r_learn = _risks(real_rows, noisy_fuser), _risks(real_rows, learned_fuser)

    def auc(pos: list[float], neg: list[float]) -> float:
        if not pos or not neg:
            return float("nan")
        return roc_auc(pos + neg, [True] * len(pos) + [False] * len(neg))

    points = []
    for tr in target_recalls:
        t_noisy = _threshold_for_recall(f_noisy, tr)
        t_learn = _threshold_for_recall(f_learn, tr)
        points.append(
            OperatingPoint(tr, _rate_at_least(r_noisy, t_noisy), _rate_at_least(r_learn, t_learn))
        )

    return FusionComparison(
        corpora=list(corpora or []),
        n_train_pos=n_pos,
        n_train_neg_synth=n_neg_synth,
        n_train_neg_real=len(real_train),
        n_test_fraud=len(fraud_rows),
        n_test_clean=len(clean_rows),
        n_test_real=len(real_rows),
        synth_auc_noisy=auc(f_noisy, c_noisy),
        synth_auc_learned=auc(f_learn, c_learn),
        real_auc_noisy=auc(f_noisy, r_noisy),
        real_auc_learned=auc(f_learn, r_learn),
        points=points,
        weights=learned.explain(),
        inactive=inactive,
    )
