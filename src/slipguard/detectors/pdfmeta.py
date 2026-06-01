"""PDF provenance detector.

Reads the original PDF off disk and flags the hard-to-fake "edited after
issuance" signals from :mod:`slipguard.forensics.pdf`: incremental updates,
editor tags in the metadata, and a modification timestamp well after creation.
Abstains on anything but a PDF with a readable ``source_path`` so it never moves
the verdict on a guess."""

from __future__ import annotations

from ..combine import noisy_or
from ..forensics.pdf import inspect_pdf
from ..models import DocumentType, FraudType, Receipt, Signal
from .base import Detector

#: per-signal P(fraud) when a signal fires; combined by noisy-OR so corroborating
#: signals compound (mirrors the verdict-level fuser).
_INCREMENTAL = 0.85
_EDITOR = 0.82
_DATE = 0.70


class PdfMetadataDetector(Detector):
    name = "pdf_meta"
    targets = frozenset({FraudType.METADATA})
    applies_to = (DocumentType.PDF,)

    def __init__(self, max_date_gap_days: float = 2.0) -> None:
        self.max_date_gap_days = max_date_gap_days

    def score(self, receipt: Receipt) -> Signal:
        if not receipt.source_path:
            return self._abstain("no source PDF on disk to inspect")
        try:
            prov = inspect_pdf(receipt.source_path)
        except OSError:
            return self._abstain("source PDF unreadable")

        parts: list[tuple[float, str]] = []
        if prov.incremental_updates > 0:
            parts.append((_INCREMENTAL,
                          f"{prov.incremental_updates} incremental update(s) appended after original write"))
        if prov.editor_tag:
            parts.append((_EDITOR, f"editor tool in metadata: {prov.editor_tag}"))
        if prov.date_gap_days is not None and prov.date_gap_days > self.max_date_gap_days:
            parts.append((_DATE, f"modified {prov.date_gap_days:.0f}d after creation"))

        evidence = {
            "incremental_updates": prov.incremental_updates,
            "producer": prov.producer,
            "creator": prov.creator,
            "editor_tag": prov.editor_tag,
            "date_gap_days": prov.date_gap_days,
        }

        if not parts:
            return Signal(self.name, 0.04, 0.85,
                          ["pdf provenance clean (single write, no editor tags, dates aligned)"],
                          evidence)

        # Corroborating provenance signals compound (same rule as the verdict fuser),
        # capped below 1.0 so PDF metadata alone is never stated as certainty.
        risk = min(0.99, noisy_or(s for s, _ in parts))
        return Signal(self.name, risk, 0.9, [r for _, r in parts], evidence)
