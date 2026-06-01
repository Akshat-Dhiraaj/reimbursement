"""Input routing — classify a raw submission so PDFs and photos take their own
provenance paths (PDF structural forensics vs. EXIF/image forensics).

Only the routing decision lives here; the route-specific extractors and
forensic detectors plug in behind it."""

from __future__ import annotations

import os
from typing import Union

from .models import DocumentType

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def route_bytes(data: bytes) -> DocumentType:
    if data[:5] == b"%PDF-":
        return DocumentType.PDF
    if data[:3] == b"\xff\xd8\xff":  # JPEG
        return DocumentType.IMAGE
    if data[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
        return DocumentType.IMAGE
    head = data[:64].lstrip()
    if head[:1] in (b"{", b"["):
        return DocumentType.STRUCTURED
    raise ValueError("unrecognised document; cannot route")


def route_path(path: Union[str, os.PathLike]) -> DocumentType:
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".pdf":
        return DocumentType.PDF
    if ext in _IMAGE_EXTS:
        return DocumentType.IMAGE
    if ext == ".json":
        return DocumentType.STRUCTURED
    with open(path, "rb") as fh:
        return route_bytes(fh.read(64))
