"""Image-provenance detector — the IMAGE-route sibling of :class:`PdfMetadataDetector`.

Aggregates two provenance sources for a receipt photo (as ``pdf_meta`` aggregates several
PDF signals), each behind its own optional extra and abstaining cleanly when absent:

* **C2PA / Content Credentials** (:mod:`slipguard.forensics.c2pa`, the ``[c2pa]`` extra) —
  *cryptographic* provenance. A signed ``trainedAlgorithmicMedia`` assertion is a
  **trustworthy positive** that the image was AI-generated/edited (the strongest IMAGE
  signal); a signed camera capture weakly exonerates.
* **EXIF** (:mod:`slipguard.forensics.image`, the ``[vlm]`` extra) — *heuristic*: an image
  *editor* named in the ``Software`` tag, or a modify timestamp well after capture.

It abstains — never accuses — whenever it cannot judge: not an IMAGE route, no source
file, neither extra installed, or **no manifest and no EXIF at all** (stripped /
screenshot / AI-generated images carry none, but so do many legitimate shared receipts,
so absent metadata is not evidence of fraud).

Calibration note: EXIF is trivially strippable and forgeable, so a *clean* read is
only weak exoneration (low score, modest confidence); a positive editor/date signal
is firmer (higher confidence), but still capped below certainty — metadata alone
never decides a verdict."""

from __future__ import annotations

from ..forensics.c2pa import c2pa_available, inspect_c2pa
from ..forensics.image import inspect_image, pillow_available
from ..models import DocumentType, FraudType, Receipt, Signal
from .base import Detector

#: per-signal P(fraud) when a signal fires; combined by noisy-OR so corroborating
#: signals compound (mirrors the verdict-level fuser and pdf_meta).
_EDITOR = 0.80
_DATE = 0.70
#: a C2PA / Content Credentials manifest that cryptographically asserts AI-generated or
#: AI-edited media — the one IMAGE signal that is a *trustworthy positive*, not a heuristic.
#: High weight, but still routed to REVIEW on its own (a single provenance defect, and we
#: have no real-world FP measurement yet); tunable upward to auto-reject.
_C2PA_AI = 0.92


class ImageMetadataDetector(Detector):
    name = "image_meta"
    targets = frozenset({FraudType.METADATA})
    applies_to = (DocumentType.IMAGE,)

    def __init__(self, max_date_gap_days: float = 2.0) -> None:
        self.max_date_gap_days = max_date_gap_days

    def score(self, receipt: Receipt) -> Signal:
        if not receipt.source_path:
            return self._abstain("no source image on disk to inspect")

        parts: list[tuple[float, str]] = []
        evidence: dict = {}
        c2pa_ai = False
        camera_signed = False

        # --- C2PA / Content Credentials: cryptographic provenance (the [c2pa] extra) ---
        if c2pa_available():
            cp = inspect_c2pa(receipt.source_path)
            if cp.has_manifest:
                evidence.update(c2pa_source_type=cp.source_type,
                                c2pa_source_uris=list(cp.source_uris),
                                c2pa_generator=cp.generator)
                if cp.source_type == "ai":
                    c2pa_ai = True
                    parts.append((_C2PA_AI, "Content Credentials assert AI-generated/edited "
                                            f"media ({cp.generator or 'unknown tool'})"))
                elif cp.source_type == "camera":
                    camera_signed = True

        # --- EXIF provenance (Pillow, the [vlm] extra) ---
        exif_judged = False
        if pillow_available():
            try:
                prov = inspect_image(receipt.source_path)
            except OSError:
                prov = None
            if prov is not None and prov.has_exif:
                exif_judged = True
                evidence.update(software=prov.software, make=prov.make, model=prov.model,
                                editor_tag=prov.editor_tag, date_gap_days=prov.date_gap_days)
                if prov.editor_tag:
                    parts.append((_EDITOR, f"image editor in EXIF Software: {prov.editor_tag}"))
                if prov.date_gap_days is not None and prov.date_gap_days > self.max_date_gap_days:
                    parts.append((_DATE, f"modified {prov.date_gap_days:.0f}d after capture"))

        # Corroborating provenance signals compound (noisy-OR, capped below certainty) — a signed
        # AI assertion is cryptographic, not heuristic -> firmer confidence than EXIF (see _fused).
        if parts:
            return self._fused(parts, 0.9 if c2pa_ai else 0.85, evidence)

        if camera_signed:
            # a cryptographically-signed camera capture exonerates more firmly than a
            # (strippable, forgeable) clean EXIF read — but still not to certainty.
            return Signal(self.name, 0.02, 0.6,
                          ["Content Credentials assert a genuine camera capture"], evidence)
        if exif_judged:
            return Signal(self.name, 0.05, 0.5,
                          ["image EXIF clean (no editor tag, capture/modify aligned)"], evidence)
        # nothing to judge: no C2PA manifest AND no EXIF. Stripped / screenshot / AI images
        # carry none, but so do many legitimate shared receipts -> abstain, never accuse.
        return self._abstain("no Content Credentials or EXIF metadata to judge provenance")
