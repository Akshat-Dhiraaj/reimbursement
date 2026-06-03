"""Date sanity — a receipt cannot be dated in the future, and one older than the reimbursement
submission window (default 60 days ~ 2 months) is out of policy and flagged for review. ``today``
is injectable so tests and audits are deterministic."""

from __future__ import annotations

from datetime import date as Date
from typing import Optional

from ..models import FraudType, Receipt, Signal
from .base import Detector


class DateSanityDetector(Detector):
    name = "date_sanity"
    targets = frozenset({FraudType.DATE})

    def __init__(self, today: Optional[Date] = None, max_age_days: int = 60):
        # max_age_days = the reimbursement submission window; a receipt older than this is
        # out of policy (default 60 days ~ 2 months). Injectable for stricter/looser policies.
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
            # Out of the reimbursement window -> flag for review (0.7 x 0.75 = 0.525 weighted,
            # past the 0.4 review threshold but below auto-reject; a late receipt may still get a
            # policy exception, so a human looks).
            return Signal(self.name, score=0.7, confidence=0.75,
                          reasons=[f"date {r.date.isoformat()} is {age}d old "
                                   f"(> {self.max_age_days}-day reimbursement window)"],
                          evidence={"age_days": age})

        return Signal(self.name, score=0.03, confidence=0.85,
                      reasons=[f"date {r.date.isoformat()} is within the {self.max_age_days}-day window"])
