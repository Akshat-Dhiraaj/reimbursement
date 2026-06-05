"""Duplicate / resubmission detection — the highest-frequency real-world abuse.

Relational, so it is ``prime``-d with prior submissions (history). Catches exact
re-submissions by (vendor, date, total) and near-duplicates via fuzzy vendor +
same date + matching amount. Perceptual-hash image matching plugs in here later
for re-photographed copies.

**Production status:** needs a persistent store of prior submissions to compare against. Until that
backend is configured it is EXCLUDED from the live path (``detectors.deployed_detectors``) and
abstains if ever run un-primed. The logic + benchmark (with a synthetic history corpus) stand ready;
a lightweight SQLite design is in ROADMAP.md / SCORECARD.md."""

from __future__ import annotations

import difflib
import re
from typing import Iterable

from ..models import FraudType, Receipt, Signal
from .base import Detector


def _norm_vendor(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class DuplicateDetector(Detector):
    name = "duplicate"
    targets = frozenset({FraudType.DUPLICATE})

    def __init__(self, amount_tol: float = 0.01, vendor_ratio: float = 0.9):
        self.amount_tol = amount_tol
        self.vendor_ratio = vendor_ratio
        self._exact: dict[tuple, str] = {}
        self._items: list[tuple[str, object, float, str]] = []

    def prime(self, history: Iterable[Receipt]) -> None:
        self._exact = {}
        self._items = []
        for r in history:
            if r.total is None:
                continue
            self._exact[self._key(r)] = r.doc_id
            self._items.append((_norm_vendor(r.vendor_name), r.date, round(r.total, 2), r.doc_id))

    def _key(self, r: Receipt) -> tuple:
        # Real extracted receipts can have an unparseable (None) date; collapse it
        # to "" so the exact-match key still hashes instead of crashing.
        date_key = r.date.isoformat() if r.date is not None else ""
        return (_norm_vendor(r.vendor_name), date_key, round(r.total or 0.0, 2))

    def score(self, receipt: Receipt) -> Signal:
        r = receipt
        if not self._exact and not self._items:
            # Not prime()-d with any prior submissions -> nothing to compare against, so ABSTAIN
            # rather than assert "no duplicate" (which would imply a check that never ran). In the
            # live product this detector is disabled entirely until a submission-history backend is
            # configured (see detectors.deployed_detectors / ROADMAP).
            return self._abstain("no submission history to compare against (needs a backend)")
        if r.total is None:
            return self._abstain("no total to match on")

        hit = self._exact.get(self._key(r))
        if hit and hit != r.doc_id:
            return Signal(self.name, score=0.95, confidence=0.9,
                          reasons=[f"exact match to prior submission {hit}"],
                          evidence={"match": hit})

        vn = _norm_vendor(r.vendor_name)
        tot = round(r.total, 2)
        for ov, od, otot, odoc in self._items:
            if odoc == r.doc_id or od != r.date:
                continue
            if abs(tot - otot) <= max(0.01, self.amount_tol * max(1.0, otot)):
                ratio = difflib.SequenceMatcher(None, vn, ov).ratio()
                if ratio >= self.vendor_ratio:
                    return Signal(self.name, score=0.88, confidence=0.8,
                                  reasons=[f"near-duplicate of {odoc} "
                                           f"(vendor~{ratio:.2f}, same date, amount~)"],
                                  evidence={"match": odoc, "vendor_ratio": round(ratio, 3)})

        return Signal(self.name, score=0.03, confidence=0.6,
                      reasons=["no matching prior submission"])
