"""Extractor registry — the canonical set, selected by document route.

Mirrors :func:`slipguard.detectors.default_detectors`: one place that lists the
available extraction approaches so the CLI and (later) an extraction benchmark can
pick or rank them. Only the dependency-free StructuredExtractor is registered
today; the OCR+KIE and VLM extractors plug in here behind the same interface."""

from __future__ import annotations

from typing import Optional

from ..models import DocumentType
from .base import Extractor
from .structured import StructuredExtractor
from .vlm_qwen import QwenVLExtractor

__all__ = [
    "Extractor", "StructuredExtractor", "QwenVLExtractor",
    "default_extractors", "image_extractors", "extractor_for",
]


def default_extractors() -> list[Extractor]:
    """The dependency-free core set used by ``score`` and the routing path. Heavy
    image extractors are kept out so importing/scoring never drags in torch."""
    return [StructuredExtractor()]


def image_extractors(model: Optional[str] = None) -> list[Extractor]:
    """Candidate IMAGE-route extractors, ranked head-to-head by ``eval-extract``.
    Construction is cheap (no model load); each declares ``available()`` so an
    un-runnable candidate is skipped with a reason rather than crashing.
    ``model`` overrides the VLM checkpoint id."""
    return [QwenVLExtractor(model_id=model) if model else QwenVLExtractor()]


def extractor_for(
    route: DocumentType, extractors: Optional[list[Extractor]] = None
) -> Optional[Extractor]:
    """First registered extractor that handles ``route``, or ``None`` if the route
    has no extractor wired yet (PDF / IMAGE until OCR/VLM lands)."""
    for ex in extractors if extractors is not None else default_extractors():
        if ex.can_handle(route):
            return ex
    return None
