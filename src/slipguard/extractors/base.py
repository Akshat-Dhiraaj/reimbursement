"""Extractor contract — turn a raw document into a normalised Receipt.

This mirrors the Detector pattern: every extraction approach (structured-JSON
passthrough now; OCR+KIE / VLM later) implements one interface, is selected by
document route, and can be benchmarked head-to-head so we pick extractors by
*measured accuracy*, not opinion — the same principle that drives detector
selection.

The real-data FP audit showed arithmetic precision is capped by extraction
quality, so extractors are first-class and separately evaluated. An extractor may
attach per-field confidence to the Receipt it returns
(:attr:`slipguard.models.Receipt.field_confidence`); the ``arithmetic`` detector
reads it to abstain on a low-confidence misread instead of asserting fraud."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import DocumentType, Receipt


class Extractor(ABC):
    #: stable identifier used in reports and config
    name: str = "extractor"
    #: document routes this approach can extract from
    handles: tuple[DocumentType, ...] = ()

    def can_handle(self, route: DocumentType) -> bool:
        return route in self.handles

    def available(self) -> tuple[bool, str]:
        """Whether this extractor can actually run here, and why not if it can't.

        Heavy extractors (OCR/VLM) override this to check their optional deps are
        importable *without* loading the model, so the benchmark can skip an
        unavailable candidate with a clear reason instead of crashing on every
        document. The dependency-free extractors are always available."""
        return True, ""

    @abstractmethod
    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        """Read the document at ``path`` and return a normalised Receipt.

        Implementations should populate ``Receipt.field_confidence`` for any field
        whose extraction is uncertain, and set ``source`` / ``source_path`` so the
        provenance detectors can run. ``doc_id`` is a fallback identifier for
        sources (a photo/PDF) that carry no id of their own."""
