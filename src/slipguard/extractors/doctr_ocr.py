"""OCR + heuristic KIE extraction via docTR — the second IMAGE-route extractor.

Why this exists: the project picks extractors by *measured* field accuracy, not
reputation, so the VLM (:mod:`slipguard.extractors.vlm_qwen`) needs at least one
head-to-head rival on the same ``eval-extract`` oracle. This is that rival, built from a
different paradigm — a two-stage OCR pipeline (text detection + recognition) followed by
the shared, transparent keyword/position KIE (:mod:`slipguard.extractors.kie`) that maps
the read lines onto the Receipt schema. No second ML model, no fine-tuning, fully
inspectable; the leaderboard says how far heuristics get versus the VLM's end-to-end
reading.

Licence (commercial-safe): docTR is Apache-2.0 and its default pretrained detection /
recognition checkpoints are Apache-2.0, and it runs on the torch the ``[vlm]`` extra
already pulls in. Heavy imports (doctr / torch) are **lazy inside the methods**, so
importing this module — and the whole package — stays dependency-free; ``available()``
reports missing deps via ``find_spec`` without loading anything.

What lives *here* (vs. the shared KIE): only the docTR-export-specific glue — reading
the OCR recogniser's per-word confidence and geometry and flattening its nested export
into the KIE's :class:`~slipguard.extractors.kie.Line` contract. Unlike a VLM, an OCR
recogniser emits a real per-word recognition confidence (softmax over the character
decoder); we surface it honestly so a garbled read makes ``arithmetic`` abstain instead
of crying fraud (the same guard the VLM arms via parse-completeness). Honest limit: it is
a *character-recognition* confidence (did we read the glyphs right), not a
field-*labelling* confidence (did we pick the right line as "total").
"""

from __future__ import annotations

from typing import Optional

from ..models import DocumentType, Receipt
from .base import Extractor, importable
from .kie import Line, receipt_from_lines


# --- OCR export -> positioned lines (docTR-specific) -------------------------

def _geom_y_center(geom: object) -> float:
    """Mean y over a docTR geometry — ``((xmin,ymin),(xmax,ymax))`` for straight pages or
    a 4-point polygon otherwise. Defensive: an unknown shape sorts to the top (y=0)."""
    if not isinstance(geom, (list, tuple)):
        return 0.0
    ys = []
    for pt in geom:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                ys.append(float(pt[1]))
            except (TypeError, ValueError):
                pass
    return sum(ys) / len(ys) if ys else 0.0


def _geom_x_min(geom: object) -> float:
    """Left edge x over a docTR geometry (box or polygon). Used only to order the
    fragments of a merged row left-to-right. Defensive: unknown shape -> 0.0 (sorts left)."""
    if not isinstance(geom, (list, tuple)):
        return 0.0
    xs = []
    for pt in geom:
        if isinstance(pt, (list, tuple)) and len(pt) >= 1:
            try:
                xs.append(float(pt[0]))
            except (TypeError, ValueError):
                pass
    return min(xs) if xs else 0.0


def _lines_from_export(export: dict) -> list[Line]:
    """Flatten docTR's exported result (pages -> blocks -> lines -> words) into KIE
    :class:`~slipguard.extractors.kie.Line`s with a vertical position and mean recognition
    confidence, sorted top-to-bottom. Pure (takes a plain dict), so the KIE layer is
    unit-testable without running OCR."""
    lines: list[Line] = []
    for page in export.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", []) or []
                text = " ".join(str(w.get("value", "")) for w in words if isinstance(w, dict)).strip()
                if not text:
                    continue
                confs = [float(w["confidence"]) for w in words
                         if isinstance(w, dict) and w.get("confidence") is not None]
                conf = sum(confs) / len(confs) if confs else 1.0
                geom = line.get("geometry")
                lines.append(Line(text=text, y=_geom_y_center(geom), conf=conf, x=_geom_x_min(geom)))
    lines.sort(key=lambda ln: ln.y)
    return lines


def _receipt_from_lines(lines: list[Line], doc_id: str, image_path: str) -> Receipt:
    """docTR is always the IMAGE route, so the read lines map onto a Receipt whose source
    is the image at ``image_path`` (the provenance detectors run image-EXIF forensics on
    it). Thin wrapper over the shared KIE so the IMAGE binding lives in one place."""
    return receipt_from_lines(
        lines, doc_id, source=DocumentType.IMAGE, source_path=image_path, image_path=image_path
    )


class DocTROCRExtractor(Extractor):
    """docTR OCR (text detection + recognition) + the shared keyword/position KIE layer
    mapping read lines onto the Receipt schema. The non-ML KIE keeps it inspectable; the
    leaderboard reports how far the heuristics get against the VLM."""

    name = "doctr"
    handles = (DocumentType.IMAGE,)

    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn") -> None:
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self._model = None

    def available(self) -> tuple[bool, str]:
        # Probe with find_spec only — never import doctr/torch here, so this stays fast
        # (the unit suite calls it) and no OCR weights are loaded.
        missing = [m for m in ("doctr", "torch", "torchvision") if not importable(m)]
        if missing:
            return False, f"missing deps: {', '.join(missing)} — pip install python-doctr"
        return True, ""

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(
            det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True
        )

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        from doctr.io import DocumentFile

        self._ensure_model()
        doc = DocumentFile.from_images(path)
        result = self._model(doc)
        lines = _lines_from_export(result.export())
        return _receipt_from_lines(lines, doc_id=doc_id or path, image_path=path)
