"""Extractor registry — the canonical set, selected by document route.

Mirrors :func:`slipguard.detectors.default_detectors`: one place that lists the
available extraction approaches so the CLI and the extraction benchmark can pick or
rank them. The lists are split by route so the dependency-free core
(:func:`default_extractors`, StructuredExtractor only) never drags in torch/pypdfium2:
the IMAGE candidates (VLM, docTR-OCR) live in :func:`image_extractors` and the PDF
candidate (born-digital text) in :func:`pdf_extractors`, all behind the same interface."""

from __future__ import annotations

from typing import Optional

from ..models import DocumentType
from .base import Extractor
from .doctr_ocr import DocTROCRExtractor
from .groq_vlm import GroqVLExtractor
from .pdf_text import PdfTextExtractor
from .structured import StructuredExtractor
from .vlm_qwen import QwenVLExtractor

__all__ = [
    "Extractor", "StructuredExtractor", "QwenVLExtractor", "DocTROCRExtractor",
    "GroqVLExtractor", "PdfTextExtractor", "default_extractors", "image_extractors",
    "pdf_extractors", "image_extractor_for_spec", "extractor_for",
]


def default_extractors() -> list[Extractor]:
    """The dependency-free core set used by ``score`` and the routing path. Heavy
    image extractors are kept out so importing/scoring never drags in torch."""
    return [StructuredExtractor()]


def pdf_extractors() -> list[Extractor]:
    """Candidate PDF-route extractors. One today (born-digital text via pypdfium2);
    construction is cheap (the pypdfium2 import is lazy) and ``available()`` reports a
    missing dep, so an un-runnable candidate is skipped with a reason, not a crash. Kept
    out of :func:`default_extractors` so the dependency-free import never needs pypdfium2."""
    return [PdfTextExtractor()]


def image_extractors(model: Optional[str] = None) -> list[Extractor]:
    """Candidate IMAGE-route extractors, ranked head-to-head by ``eval-extract``: the
    end-to-end VLM and the OCR+KIE pipeline, two different paradigms on the same oracle.
    Construction is cheap (no model load); each declares ``available()`` so an un-runnable
    candidate is skipped with a reason rather than crashing. The VLM is listed first so it
    stays ``score``'s default until the leaderboard names a winner. ``model`` overrides the
    VLM checkpoint id (docTR ignores it — it picks det/reco arches, not an HF id)."""
    return [
        QwenVLExtractor(model_id=model) if model else QwenVLExtractor(),
        DocTROCRExtractor(),
    ]


def image_extractor_for_spec(spec: str) -> Extractor:
    """Resolve an ``eval-real --extractor`` spec to exactly ONE image extractor: ``doctr``
    -> the OCR+KIE pipeline, ``vlm`` -> the default VLM checkpoint, anything else -> a VLM
    on that HF checkpoint id. (eval-real audits one extractor at a time; using a spec map —
    not "first runnable" over :func:`image_extractors` — means ``--extractor vlm`` returns
    the VLM rather than whichever extractor happens to come first in the list.)"""
    if spec == "doctr":
        return DocTROCRExtractor()
    if spec == "vlm":
        return QwenVLExtractor()
    if spec == "groq":
        return GroqVLExtractor()
    if spec.startswith("groq:"):  # groq:<model-id> selects a specific Groq model
        return GroqVLExtractor(model_id=spec[len("groq:"):])
    return QwenVLExtractor(model_id=spec)


def extractor_for(
    route: DocumentType, extractors: Optional[list[Extractor]] = None
) -> Optional[Extractor]:
    """First extractor in ``extractors`` (default: the dependency-free core set) that
    handles ``route``, or ``None`` if none does. The default set is STRUCTURED-only, so
    callers wanting the IMAGE/PDF routes pass :func:`image_extractors` / :func:`pdf_extractors`
    (or ``score`` falls back to them) to avoid importing torch/pypdfium2 unless needed."""
    for ex in extractors if extractors is not None else default_extractors():
        if ex.can_handle(route):
            return ex
    return None
