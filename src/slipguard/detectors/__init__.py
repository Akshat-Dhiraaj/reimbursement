"""Detection approaches. ``default_detectors()`` is the canonical set the harness
ranks; new approaches (VLM-consistency, AI-image, tamper-localisation, PDF /
metadata forensics) are added here behind the same ``Detector`` contract."""

from __future__ import annotations

from .arithmetic import ArithmeticConsistencyDetector
from .base import Detector
from .datesanity import DateSanityDetector
from .duplicate import DuplicateDetector
from .pdfmeta import PdfMetadataDetector
from .taxid import TaxIdValidationDetector


def default_detectors() -> list[Detector]:
    return [
        ArithmeticConsistencyDetector(),
        TaxIdValidationDetector(),
        DateSanityDetector(),
        DuplicateDetector(),
        PdfMetadataDetector(),
    ]


__all__ = [
    "Detector",
    "ArithmeticConsistencyDetector",
    "TaxIdValidationDetector",
    "DateSanityDetector",
    "DuplicateDetector",
    "PdfMetadataDetector",
    "default_detectors",
]
