"""Signal fusion -> calibrated risk + decision.

Baseline noisy-OR over confidence-weighted signals: independent fraud signals
compound, and abstaining detectors (confidence 0) cannot move the score. It needs
no training and its rule is obvious, so it is the default.

A ``combiner`` can override the rule with a learned/calibrated function over the
same signals (see :class:`slipguard.fusion_learned.LearnedFuser`); thresholds,
:meth:`decide` and :meth:`verdict` are shared so only the score combination
differs. Selection between the two is by measured numbers (``eval-fusion``), not
by preference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .combine import noisy_or
from .models import Decision, Signal, Verdict


@dataclass
class Fuser:
    review_threshold: float = 0.4
    reject_threshold: float = 0.85
    #: optional learned combination over the signals; ``None`` -> noisy-OR baseline.
    combiner: Optional[Callable[[Sequence[Signal]], float]] = None

    def risk(self, signals: Iterable[Signal]) -> float:
        signals = list(signals)
        if self.combiner is not None:
            # A learned combiner reads every signal (it zeroes abstainers itself via
            # their weighted==0); clamp so a miscalibrated model can't escape [0, 1].
            return max(0.0, min(1.0, self.combiner(signals)))
        # Baseline: combine each detector's confidence-weighted score; abstainers
        # (weighted 0) are dropped so an irrelevant detector can't move the verdict.
        return noisy_or(s.weighted for s in signals if not s.abstained)

    def decide(self, risk: float) -> Decision:
        if risk >= self.reject_threshold:
            return Decision.REJECT
        if risk >= self.review_threshold:
            return Decision.REVIEW
        return Decision.APPROVE

    def verdict(self, doc_id: str, signals: Iterable[Signal]) -> Verdict:
        signals = list(signals)
        risk = self.risk(signals)
        return Verdict(doc_id=doc_id, risk_score=risk, decision=self.decide(risk), signals=signals)
