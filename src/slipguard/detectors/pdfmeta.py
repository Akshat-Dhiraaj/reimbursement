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

from ..forensics.pdf import inspect_pdf, inspect_pdf_deep
from ..models import DocumentType, FraudType, Receipt, Signal
from .base import Detector

#: per-signal P(fraud) when a signal fires; combined by noisy-OR so corroborating
#: signals compound (mirrors the verdict-level fuser).
_INCREMENTAL = 0.85
#: an incremental update that rewrote a page CONTENT stream (the displayed values), not
#: just metadata. More specific than _INCREMENTAL: legitimate incremental updates exist
#: (digital signatures, form fills) but they don't rewrite the page content, so a content
#: rewrite after issuance is a strong "the visible amount was edited" signal -> slightly
#: above the generic incremental weight. The two compound when both fire.
_CONTENT_EDIT = 0.88
#: bytes appended after a digital signature's /ByteRange — content added/edited AFTER the
#: document was signed. Strong (a signed document should be final), but a single defect, so
#: it routes to REVIEW on its own and compounds with any incremental/content signal it
#: genuinely co-occurs with (an edit-after-signing is usually also an incremental update).
_SIG_EDIT = 0.80
_EDITOR = 0.82
_DATE = 0.70
#: structural risks (deep layer). JS / auto-run actions never appear in a genuine
#: issued receipt -> strong; a fillable form or an overlay box is suspicious but has
#: legitimate cousins (signed PDFs, form-style invoices) -> weaker, to keep FP low.
_JAVASCRIPT = 0.80
_OPEN_ACTION = 0.75
_ACROFORM = 0.45
_OVERLAY = 0.50
#: a cover-and-relabel overlay drawn IN the page content stream (a white rectangle over
#: pre-existing text, then a new value on top) — rarer/less legitimate than an overlay
#: *annotation*, so slightly higher, but still suspicious-not-proof -> REVIEW.
_CONTENT_OVERLAY = 0.55


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
        # Disjoint accounting: a revision already explained as a content edit or as an
        # edit-after-signing is NOT also counted as a generic incremental update — noisy-OR
        # of two views of one revision would manufacture a spurious over-escalation (the
        # more-specific signal already carries the higher weight). _INCREMENTAL covers only
        # the appended revisions we did not otherwise localize.
        sig_updates = 1 if prov.signature_uncovered_bytes > 0 else 0
        other_incremental = max(0, prov.incremental_updates - prov.content_stream_edits - sig_updates)
        if other_incremental > 0:
            parts.append((_INCREMENTAL,
                          f"{other_incremental} incremental update(s) appended after original write"))
        if prov.content_stream_edits > 0:
            parts.append((_CONTENT_EDIT,
                          "incremental update rewrote the page content stream "
                          "(displayed values edited after issuance)"))
        if prov.signature_uncovered_bytes > 0:
            parts.append((_SIG_EDIT,
                          f"{prov.signature_uncovered_bytes} byte(s) appended after the "
                          "digital signature was applied (edited after signing)"))
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
            if deep.content_overlays > 0:
                parts.append((_CONTENT_OVERLAY,
                              f"{deep.content_overlays} content-stream overlay(s) covering "
                              "existing text (cover-and-relabel)"))

        evidence = {
            "incremental_updates": prov.incremental_updates,
            "content_stream_edits": prov.content_stream_edits,
            "is_signed": prov.is_signed,
            "signature_uncovered_bytes": prov.signature_uncovered_bytes,
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
                content_overlays=deep.content_overlays,
            )

        if not parts:
            return Signal(self.name, 0.04, 0.85,
                          ["pdf provenance clean (single write, no editor tags, dates aligned)"],
                          evidence)
        # Corroborating provenance signals compound (noisy-OR, capped below certainty) — see _fused.
        return self._fused(parts, 0.9, evidence)
