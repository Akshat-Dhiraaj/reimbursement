"""Provenance forensics on the original document (PDF structure/metadata, later
EXIF). These read the file as shipped — signals that survive field extraction and
are hard to fake without leaving a trace."""

from .pdf import (
    KNOWN_EDITORS,
    DeepPdfProvenance,
    PdfProvenance,
    inspect_pdf,
    inspect_pdf_deep,
    pikepdf_available,
)

__all__ = [
    "PdfProvenance", "inspect_pdf", "KNOWN_EDITORS",
    "DeepPdfProvenance", "inspect_pdf_deep", "pikepdf_available",
]
