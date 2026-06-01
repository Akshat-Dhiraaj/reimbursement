"""Date sanity — a receipt cannot be dated in the future, and very old dates are
mildly suspicious. ``today`` is injectable so tests are deterministic."""

from __future__ import annotations

from datetime import date as Date
from typing import Optional

from ..models import FraudType, Receipt, Signal
from .base import Detector


class DateSanityDetector(Detector):
    name = "date_sanity"
    targets = frozenset({FraudType.DATE})

    def __init__(self, today: Optional[Date] = None, max_age_days: int = 1825):
        self._today = today
        self.max_age_days = max_age_days

    @property
    def today(self) -> Date:
        return self._today or Date.today()

    def score(self, receipt: Receipt) -> Signal:
        r = receipt
        if r.date is None:
            return self._abstain("no date")
        today = self.today

        if r.date > today:
            days = (r.date - today).days
            return Signal(self.name, score=0.92, confidence=0.9,
                          reasons=[f"date {r.date.isoformat()} is in the future "
                                   f"(+{days}d vs {today.isoformat()})"],
                          evidence={"days_future": days})

        age = (today - r.date).days
        if age > self.max_age_days:
            return Signal(self.name, score=0.6, confidence=0.6,
                          reasons=[f"date {r.date.isoformat()} is {age}d old "
                                   f"(> {self.max_age_days})"],
                          evidence={"age_days": age})

        return Signal(self.name, score=0.03, confidence=0.85,
                      reasons=[f"date {r.date.isoformat()} is plausible"])
