"""Provenance forensics on the original document (PDF structure/metadata, later
EXIF). These read the file as shipped — signals that survive field extraction and
are hard to fake without leaving a trace."""

from .pdf import KNOWN_EDITORS, PdfProvenance, inspect_pdf

__all__ = ["PdfProvenance", "inspect_pdf", "KNOWN_EDITORS"]
