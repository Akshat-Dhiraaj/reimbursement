"""Evaluation harness — runs every approach independently on the labelled
benchmark and ranks them (overall AUC, per-subtype recall, false-positive rate),
plus the fused verdict. This is the empirical-selection engine."""

from __future__ import annotations

from .harness import DetectorReport, FusionReport, Report, evaluate

__all__ = ["evaluate", "Report", "DetectorReport", "FusionReport"]
