"""Detection approaches. ``default_detectors()`` is the canonical set the harness
ranks; new approaches (VLM-consistency, AI-image, tamper-localisation, PDF /
metadata forensics) are added here behind the same ``Detector`` contract."""

from __future__ import annotations

from .arithmetic import ArithmeticConsistencyDetector
from .base import Detector
from .datesanity import DateSanityDetector
from .duplicate import DuplicateDetector
from .imagemeta import ImageMetadataDetector
from .pdfmeta import PdfMetadataDetector
from .taxid import TaxIdValidationDetector


def default_detectors() -> list[Detector]:
    """The full detector roster, used by the BENCHMARK (which ``prime``-s the relational ones with a
    history corpus). For the live single-document product, use :func:`deployed_detectors`."""
    return [
        ArithmeticConsistencyDetector(),
        TaxIdValidationDetector(),
        DateSanityDetector(),
        DuplicateDetector(),   # RELATIONAL: only meaningful once prime()-d with prior submissions
        PdfMetadataDetector(),
        ImageMetadataDetector(),
    ]


#: Detectors that are RELATIONAL — they compare a receipt against PRIOR submissions, so they need a
#: persistent submission-history store to do anything. Excluded from the live product path until that
#: backend is configured (running them un-primed would surface a check that can never fire). See
#: ROADMAP.md / SCORECARD.md for the lightweight SQLite design.
_NEEDS_HISTORY_BACKEND = frozenset({"duplicate"})


def deployed_detectors() -> list[Detector]:
    """The detectors that actually work in the live, single-document product path (web UI /
    ``validate`` / ``score``), which carries no prior-submission history. Excludes the relational
    detectors in :data:`_NEEDS_HISTORY_BACKEND` (duplicate / resubmission) — DISABLED until a
    persistent submission store is wired. The logic and its benchmark live on."""
    return [d for d in default_detectors() if d.name not in _NEEDS_HISTORY_BACKEND]


__all__ = [
    "Detector",
    "ArithmeticConsistencyDetector",
    "TaxIdValidationDetector",
    "DateSanityDetector",
    "DuplicateDetector",
    "PdfMetadataDetector",
    "ImageMetadataDetector",
    "default_detectors",
    "deployed_detectors",
]
