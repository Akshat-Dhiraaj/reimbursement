"""Arithmetic consistency — reconcile line items -> subtotal -> tax -> total.

The benchmarks (GPT4o-Receipt, AIForge-Doc) found this is the single most
reliable signal against AI-fabricated receipts and, unlike pixel forensics, it
survives screenshots and recompression because it reads content, not artifacts.
"""

from __future__ import annotations

from ..models import FraudType, Receipt, Signal
from .base import Detector


class ArithmeticConsistencyDetector(Detector):
    name = "arithmetic"
    targets = frozenset({FraudType.ARITHMETIC})

    def __init__(self, rel_tol: float = 0.01, abs_tol: float = 0.02):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def _err(self, printed: float, expected: float) -> tuple[bool, float]:
        tol = max(self.abs_tol, self.rel_tol * abs(expected))
        diff = abs(printed - expected)
        return diff > tol, diff / max(1.0, abs(expected))

    def score(self, receipt: Receipt) -> Signal:
        r = receipt
        failures: list[tuple[str, float]] = []
        have_items = bool(r.line_items)

        computed_subtotal = 0.0
        for i, li in enumerate(r.line_items):
            expected = round(li.quantity * li.unit_price, 2)
            computed_subtotal += li.amount
            bad, rel = self._err(li.amount, expected)
            if bad:
                failures.append(
                    (f"line[{i}] '{li.description}': amount {li.amount} != qty*price {expected}", rel)
                )
        computed_subtotal = round(computed_subtotal, 2)

        ref_subtotal = r.subtotal if r.subtotal is not None else (computed_subtotal if have_items else None)

        if r.subtotal is not None and have_items:
            bad, rel = self._err(r.subtotal, computed_subtotal)
            if bad:
                failures.append((f"subtotal {r.subtotal} != sum(line items) {computed_subtotal}", rel))

        if r.tax_rate is not None and ref_subtotal is not None and r.tax_amount is not None:
            expected_tax = round(r.tax_rate * ref_subtotal, 2)
            bad, rel = self._err(r.tax_amount, expected_tax)
            if bad:
                failures.append((f"tax {r.tax_amount} != rate*subtotal {expected_tax}", rel))

        if r.total is not None and ref_subtotal is not None:
            expected_total = round(ref_subtotal + (r.tax_amount or 0.0), 2)
            bad, rel = self._err(r.total, expected_total)
            if bad:
                failures.append((f"total {r.total} != subtotal+tax {expected_total}", rel))

        can_reconcile = (ref_subtotal is not None and r.total is not None) or have_items
        if not can_reconcile:
            return self._abstain("insufficient fields to reconcile")

        evidence = {"computed_subtotal": computed_subtotal}
        if not failures:
            return Signal(self.name, score=0.03, confidence=0.9,
                          reasons=["all arithmetic reconciles"], evidence=evidence)

        max_rel = max(rel for _, rel in failures)
        score = min(0.99, 0.6 + 4.0 * max_rel)
        evidence["max_rel_error"] = round(max_rel, 4)
        return Signal(self.name, score=score, confidence=0.92,
                      reasons=[msg for msg, _ in failures], evidence=evidence)
