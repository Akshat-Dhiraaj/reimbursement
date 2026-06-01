"""Image EXIF provenance detector — the IMAGE-route sibling of :class:`PdfMetadataDetector`.

Reads the receipt photo off disk and flags the hard-to-fake "edited after capture"
signals from :mod:`slipguard.forensics.image`: an image *editor* named in the EXIF
``Software`` tag, and a modify timestamp well after the capture timestamp. It
abstains — never accuses — whenever it cannot judge: not an IMAGE route, no source
file, Pillow unavailable, or **no EXIF at all** (stripped/screenshot/AI-generated
images are common and legitimate, so absent metadata is not evidence of fraud).

Calibration note: EXIF is trivially strippable and forgeable, so a *clean* read is
only weak exoneration (low score, modest confidence); a positive editor/date signal
is firmer (higher confidence), but still capped below certainty — metadata alone
never decides a verdict."""

from __future__ import annotations

from ..combine import noisy_or
from ..forensics.image import inspect_image, pillow_available
from ..models import DocumentType, FraudType, Receipt, Signal
from .base import Detector

#: per-signal P(fraud) when a signal fires; combined by noisy-OR so corroborating
#: signals compound (mirrors the verdict-level fuser and pdf_meta).
_EDITOR = 0.80
_DATE = 0.70


class ImageMetadataDetector(Detector):
    name = "image_meta"
    targets = frozenset({FraudType.METADATA})
    applies_to = (DocumentType.IMAGE,)

    def __init__(self, max_date_gap_days: float = 2.0) -> None:
        self.max_date_gap_days = max_date_gap_days

    def score(self, receipt: Receipt) -> Signal:
        if not receipt.source_path:
            return self._abstain("no source image on disk to inspect")
        if not pillow_available():
            return self._abstain('Pillow not installed — pip install -e ".[vlm]"')
        try:
            prov = inspect_image(receipt.source_path)
        except OSError:
            return self._abstain("source image unreadable")
        if not prov.has_exif:
            # stripped / screenshot / AI-generated images carry no EXIF, but so do many
            # legitimate shared receipts — absent metadata is not evidence of fraud.
            return self._abstain("no EXIF metadata to judge provenance (stripped/screenshot/AI)")

        parts: list[tuple[float, str]] = []
        if prov.editor_tag:
            parts.append((_EDITOR, f"image editor in EXIF Software: {prov.editor_tag}"))
        if prov.date_gap_days is not None and prov.date_gap_days > self.max_date_gap_days:
            parts.append((_DATE, f"modified {prov.date_gap_days:.0f}d after capture"))

        evidence = {
            "software": prov.software,
            "make": prov.make,
            "model": prov.model,
            "editor_tag": prov.editor_tag,
            "date_gap_days": prov.date_gap_days,
        }

        if not parts:
            # EXIF present and undefective — but strippable/forgeable, so this exonerates
            # only weakly (modest confidence), unlike a born-clean structured receipt.
            return Signal(self.name, 0.05, 0.5,
                          ["image EXIF clean (no editor tag, capture/modify aligned)"],
                          evidence)

        # Corroborating provenance signals compound (same rule as the verdict fuser),
        # capped below 1.0 so image metadata alone is never stated as certainty.
        risk = min(0.99, noisy_or(s for s, _ in parts))
        return Signal(self.name, risk, 0.85, [r for _, r in parts], evidence)
