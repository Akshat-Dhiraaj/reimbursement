"""PDF provenance detector.

Reads the original PDF off disk and flags the hard-to-fake "edited after
issuance" signals from :mod:`slipguard.forensics.pdf`. It uses **both** layers of
that inspector:

* **Layer 1 (always):** the dependency-free byte scan — incremental updates
  (extra ``%%EOF``), editor tags, and a ModDate well after CreationDate.
* **Layer 2 (when the ``[pdf-forensics]`` extra is installed):** pikepdf, which
  **recovers the editor tag / date gap on modern compressed PDFs** that Layer 1
  can't read (the Info dict is in an object stream / only in XMP), and adds
  structural anomalies — a fillable AcroForm, embedded JavaScript, an auto-run
  OpenAction, or cover-and-relabel annotation overlays.

Abstains on anything but a PDF with a readable ``source_path`` so it never moves
the verdict on a guess."""

from __future__ import annotations

from ..combine import noisy_or
from ..forensics.pdf import inspect_pdf, inspect_pdf_deep
from ..models import DocumentType, FraudType, Receipt, Signal
from .base import Detector

#: per-signal P(fraud) when a signal fires; combined by noisy-OR so corroborating
#: signals compound (mirrors the verdict-level fuser).
_INCREMENTAL = 0.85
_EDITOR = 0.82
_DATE = 0.70
#: structural risks (deep layer). JS / auto-run actions never appear in a genuine
#: issued receipt -> strong; a fillable form or an overlay box is suspicious but has
#: legitimate cousins (signed PDFs, form-style invoices) -> weaker, to keep FP low.
_JAVASCRIPT = 0.80
_OPEN_ACTION = 0.75
_ACROFORM = 0.45
_OVERLAY = 0.50


class PdfMetadataDetector(Detector):
    name = "pdf_meta"
    targets = frozenset({FraudType.METADATA})
    applies_to = (DocumentType.PDF,)

    def __init__(self, max_date_gap_days: float = 2.0, use_deep: bool = True) -> None:
        self.max_date_gap_days = max_date_gap_days
        #: run the pikepdf deep layer when available. Off forces the dependency-free
        #: byte path even with the extra installed — the ``eval-pdf-forensics`` benchmark
        #: flips this to measure Layer 2's marginal recall (byte-only vs deep) on the same
        #: corpus, and it is an escape hatch to disable the heavier path without uninstalling.
        self.use_deep = use_deep

    def score(self, receipt: Receipt) -> Signal:
        if not receipt.source_path:
            return self._abstain("no source PDF on disk to inspect")
        try:
            prov = inspect_pdf(receipt.source_path)
        except OSError:
            return self._abstain("source PDF unreadable")
        # None if disabled, extra absent, or file unreadable by pikepdf
        deep = inspect_pdf_deep(receipt.source_path) if self.use_deep else None

        # The editor tag and date gap come from Layer 1, falling back to Layer 2's
        # decoded metadata when the byte scan saw nothing (the compressed-PDF case).
        editor_tag = prov.editor_tag or (deep.editor_tag if deep else None)
        date_gap = prov.date_gap_days
        if date_gap is None and deep is not None:
            date_gap = deep.date_gap_days

        parts: list[tuple[float, str]] = []
        if prov.incremental_updates > 0:
            parts.append((_INCREMENTAL,
                          f"{prov.incremental_updates} incremental update(s) appended after original write"))
        if editor_tag:
            parts.append((_EDITOR, f"editor tool in metadata: {editor_tag}"))
        if date_gap is not None and date_gap > self.max_date_gap_days:
            parts.append((_DATE, f"modified {date_gap:.0f}d after creation"))
        if deep is not None:
            if deep.has_javascript:
                parts.append((_JAVASCRIPT, "embedded JavaScript (receipts never carry script)"))
            if deep.has_open_action:
                parts.append((_OPEN_ACTION, "auto-run /OpenAction on open"))
            if deep.has_acroform:
                parts.append((_ACROFORM, "rendered as a fillable AcroForm, not a flat receipt"))
            if deep.overlay_annotations > 0:
                parts.append((_OVERLAY,
                              f"{deep.overlay_annotations} overlay annotation(s) that can cover values"))

        evidence = {
            "incremental_updates": prov.incremental_updates,
            "producer": prov.producer or (deep.producer if deep else None),
            "creator": prov.creator or (deep.creator if deep else None),
            "editor_tag": editor_tag,
            "date_gap_days": date_gap,
            "deep_layer": deep is not None,
        }
        if deep is not None:
            evidence.update(
                pdf_version=deep.pdf_version,
                xmp_creator_tool=deep.xmp_creator_tool,
                xmp_history_events=deep.xmp_history_events,
                has_acroform=deep.has_acroform,
                has_javascript=deep.has_javascript,
                has_open_action=deep.has_open_action,
                overlay_annotations=deep.overlay_annotations,
            )

        if not parts:
            return Signal(self.name, 0.04, 0.85,
                          ["pdf provenance clean (single write, no editor tags, dates aligned)"],
                          evidence)

        # Corroborating provenance signals compound (same rule as the verdict fuser),
        # capped below 1.0 so PDF metadata alone is never stated as certainty.
        risk = min(0.99, noisy_or(s for s, _ in parts))
        return Signal(self.name, risk, 0.9, [r for _, r in parts], evidence)
