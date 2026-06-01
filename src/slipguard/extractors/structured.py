"""Structured-JSON extractor — the trivial, dependency-free reference Extractor.

The STRUCTURED route is already field-level data, so "extraction" is just loading
and normalising the JSON into a Receipt. It exists so the score pipeline runs
through the same Extractor -> detectors path for every route, and so the interface
has a working implementation to test against before the OCR/VLM extractors land."""

from __future__ import annotations

import json
from typing import Optional

from ..models import DocumentType, Receipt
from .base import Extractor


class StructuredExtractor(Extractor):
    name = "structured"
    handles = (DocumentType.STRUCTURED,)

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        with open(path, "r", encoding="utf-8") as fh:
            receipt = Receipt.from_dict(json.load(fh))
        if doc_id and not receipt.doc_id:
            receipt.doc_id = doc_id
        return receipt
