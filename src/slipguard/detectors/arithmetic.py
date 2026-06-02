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

    def __init__(self, rel_tol: float = 0.01, abs_tol: float = 0.02, min_confidence: float = 0.5):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        # Abstain when the money fields were extracted below this confidence (see
        # _extraction_confidence). Dormant for receipts with no confidence info.
        self.min_confidence = min_confidence

    def _err(self, printed: float, expected: float) -> tuple[bool, float]:
        # Tolerance = a flat floor (rounding noise on small amounts) OR a relative
        # band (whichever is larger), so neither cent-level rounding nor large
        # legitimate totals trip a false mismatch.
        tol = max(self.abs_tol, self.rel_tol * abs(expected))
        diff = abs(printed - expected)
        # Second value is the *normalised* error used later to scale severity;
        # max(1.0, ...) keeps a tiny `expected` from exploding the ratio.
        return diff > tol, diff / max(1.0, abs(expected))

    def _extraction_confidence(self, r: Receipt) -> float:
        """Lowest extraction confidence among the money fields this detector reads.
        An empty ``field_confidence`` (synthetic, hand-written, or oracle receipts)
        means "no confidence info" and is treated as fully trusted -> 1.0."""
        fc = r.field_confidence
        if not fc:
            return 1.0
        keys = []
        if r.subtotal is not None:
            keys.append("subtotal")
        if r.tax_amount is not None:
            keys.append("tax_amount")
        if r.total is not None:
            keys.append("total")
        if r.line_items:
            keys.append("line_items")
        present = [fc[k] for k in keys if k in fc]
        return min(present) if present else 1.0

    def score(self, receipt: Receipt) -> Signal:
        r = receipt

        # Extraction guard: if the money fields were read with low confidence, a
        # "mismatch" is more likely a misread than fraud — abstain instead of
        # crying wolf (the audit's recommended fix). Only bites once a real
        # extractor reports low confidence; trusted fields read as 1.0.
        conf = self._extraction_confidence(r)
        if conf < self.min_confidence:
            return self._abstain(
                f"extraction confidence {conf:.2f} < {self.min_confidence:.2f}; not asserting arithmetic"
            )

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

        # Use the printed subtotal as the reference; if none was extracted, fall
        # back to the sum of line items so the tax/total checks can still run.
        ref_subtotal = r.subtotal if r.subtotal is not None else (computed_subtotal if have_items else None)

        if r.subtotal is not None and have_items:
            bad, rel = self._err(r.subtotal, computed_subtotal)
            # Tax-inclusive line prices: the lines legitimately total subtotal + tax (not the
            # pre-tax subtotal) — common on Indonesian/EU menus. Only flag if NEITHER the
            # tax-exclusive nor the tax-inclusive interpretation reconciles, so a genuine
            # tax-inclusive receipt is not a false positive (the CORD subtotal!=Σlines cases).
            if bad and r.tax_amount is not None:
                bad, _ = self._err(r.subtotal + r.tax_amount, computed_subtotal)
            if bad:
                failures.append((f"subtotal {r.subtotal} != sum(line items) {computed_subtotal}", rel))

        if r.tax_rate is not None and ref_subtotal is not None and r.tax_amount is not None:
            expected_tax = round(r.tax_rate * ref_subtotal, 2)
            bad, rel = self._err(r.tax_amount, expected_tax)
            if bad:
                failures.append((f"tax {r.tax_amount} != rate*subtotal {expected_tax}", rel))

        if r.total is not None and ref_subtotal is not None:
            # total reconciles against subtotal + tax + service charge - discount. The
            # service/discount terms default to 0 when absent, so a plain 3-field receipt is
            # unchanged; a receipt that legitimately carries them no longer trips (the CORD
            # total!=subtotal+tax cases, the measured FP this richer model was added to fix).
            expected_total = round(ref_subtotal + (r.tax_amount or 0.0)
                                   + (r.service_charge or 0.0) - (r.discount or 0.0), 2)
            bad, rel = self._err(r.total, expected_total)
            if bad:
                label = "subtotal+tax+service-discount" if (r.service_charge or r.discount) else "subtotal+tax"
                failures.append((f"total {r.total} != {label} {expected_total}", rel))

        # We can only judge if there is something to cross-check: a subtotal+total
        # pair, or line items. With neither, abstain rather than guess — this guard
        # is what stops the detector crying wolf on a lossily-extracted receipt.
        can_reconcile = (ref_subtotal is not None and r.total is not None) or have_items
        if not can_reconcile:
            return self._abstain("insufficient fields to reconcile")

        evidence = {"computed_subtotal": computed_subtotal}
        if not failures:
            return Signal(self.name, score=0.03, confidence=0.9,
                          reasons=["all arithmetic reconciles"], evidence=evidence)

        # Any failure starts at 0.6 ("probably fraud") and ramps with the worst
        # normalised error, capped at 0.99 so it's never stated as certainty.
        max_rel = max(rel for _, rel in failures)
        score = min(0.99, 0.6 + 4.0 * max_rel)
        evidence["max_rel_error"] = round(max_rel, 4)
        return Signal(self.name, score=score, confidence=0.92,
                      reasons=[msg for msg, _ in failures], evidence=evidence)
