"""Detector contract. Every detection approach implements this so the harness
can run them independently and rank them on measured performance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ..models import DocumentType, FraudType, Receipt, Signal


class Detector(ABC):
    #: stable identifier used in reports and config
    name: str = "detector"
    #: fraud subtypes this approach is built to catch (for per-subtype scoring)
    targets: frozenset[FraudType] = frozenset()
    #: document routes this approach can score; ``None`` means any
    applies_to: Optional[tuple[DocumentType, ...]] = None

    def applicable(self, receipt: Receipt) -> bool:
        if self.applies_to is None:
            return True
        return receipt.source in self.applies_to

    def prime(self, history: Iterable[Receipt]) -> None:
        """Optionally seed the detector with prior submissions (e.g. for
        duplicate detection). Stateless detectors leave this as a no-op."""

    @abstractmethod
    def score(self, receipt: Receipt) -> Signal:
        """Return a calibrated fraud signal for one receipt."""

    def run(self, receipt: Receipt) -> Signal:
        """Score the receipt, or abstain if this approach doesn't apply to its route.
        Shared entry point used by the harness and the CLI."""
        if not self.applicable(receipt):
            return self._abstain("not applicable for this document type")
        return self.score(receipt)

    def _abstain(self, reason: str) -> Signal:
        return Signal(detector=self.name, score=0.0, confidence=0.0, reasons=[reason])
