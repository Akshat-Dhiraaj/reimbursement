"""Benchmark data. The synthetic generator mints license-clean labelled
real/fraud pairs with field-level ground truth; real corpora (Find-it-again,
DocTamper, AIForge-Doc) plug in alongside it as loaders later."""

from __future__ import annotations

from .synth import Dataset, generate

__all__ = ["Dataset", "generate"]
