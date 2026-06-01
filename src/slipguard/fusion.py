"""Signal fusion -> calibrated risk + decision.

Baseline noisy-OR over confidence-weighted signals: independent fraud signals
compound, and abstaining detectors (confidence 0) cannot move the score. This is
deliberately simple and replaceable by a learned/calibrated fuser once the
harness gives us labelled per-detector performance to fit on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import Decision, Signal, Verdict


@dataclass
class Fuser:
    review_threshold: float = 0.4
    reject_threshold: float = 0.85

    def risk(self, signals: Iterable[Signal]) -> float:
        prod = 1.0
        for s in signals:
            if s.abstained:
                continue
            prod *= 1.0 - max(0.0, min(1.0, s.weighted))
        return 1.0 - prod

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
