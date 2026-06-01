"""Probability combination — the one noisy-OR rule, defined once.

Noisy-OR treats each input as an independent probability that fraud is present
and asks "what's the chance at least one of them is right": ``1 - Π(1 - pᵢ)``.
It is used in two places — across detectors at the verdict level
(:mod:`slipguard.fusion`) and across the PDF detector's own provenance signals
(:mod:`slipguard.detectors.pdfmeta`) — so the formula lives here instead of being
re-derived (and independently bug-fixed) in each spot."""

from __future__ import annotations

from typing import Iterable


def noisy_or(probabilities: Iterable[float]) -> float:
    """Combine independent fraud probabilities by noisy-OR.

    Each value is clamped to [0, 1] so a stray out-of-range score can't push the
    running product negative or the result above 1. An empty input means "nobody
    spoke" and returns 0.0 (the product of no factors is 1, so ``1 - 1``)."""
    prod = 1.0
    for p in probabilities:
        prod *= 1.0 - max(0.0, min(1.0, p))
    return 1.0 - prod
