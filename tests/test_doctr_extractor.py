"""Tests for the docTR OCR extractor's *own* parts — flattening the OCR export (nested
pages->blocks->lines->words, with per-word recognition confidence and geometry) into the
shared KIE's ``Line`` contract, and the IMAGE-route binding — with no model load and no
doctr/torch imported. The paradigm-agnostic KIE itself is tested in ``test_kie.py``.

The final block proves the genuine OCR recognition confidence is wired through to the
arithmetic abstain guard, exactly as the VLM's parse-completeness confidence is.
"""

from __future__ import annotations

from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.extractors import (
    DocTROCRExtractor,
    QwenVLExtractor,
    image_extractor_for_spec,
    image_extractors,
)
from slipguard.extractors.doctr_ocr import (
    _geom_x_min,
    _geom_y_center,
    _lines_from_export,
    _receipt_from_lines,
)
from slipguard.extractors.kie import Line
from slipguard.models import DocumentType


def _ln(text: str, y: float = 0.5, conf: float = 1.0, x: float = 0.0) -> Line:
    return Line(text=text, y=y, conf=conf, x=x)


# --- OCR export -> lines -----------------------------------------------------

def test_geom_y_center_box_and_polygon():
    assert _geom_y_center(((0.0, 0.1), (1.0, 0.3))) == 0.2          # straight box
    assert _geom_y_center([(0, 0.1), (1, 0.1), (1, 0.3), (0, 0.3)]) == 0.2  # 4-pt polygon
    assert _geom_y_center("bogus") == 0.0                           # defensive -> top


def test_geom_x_min_box_and_polygon():
    assert _geom_x_min(((0.6, 0.1), (0.9, 0.3))) == 0.6           # straight box left edge
    assert _geom_x_min([(0.6, 0.1), (0.9, 0.1), (0.9, 0.3), (0.6, 0.3)]) == 0.6  # polygon
    assert _geom_x_min("bogus") == 0.0                            # defensive -> left


def test_lines_from_export_flattens_joins_and_sorts():
    export = {
        "pages": [{
            "blocks": [{
                "lines": [
                    {  # printed lower on the page, but listed first
                        "geometry": ((0.0, 0.80), (1.0, 0.84)),
                        "words": [
                            {"value": "TOTAL", "confidence": 1.0},
                            {"value": "12.00", "confidence": 0.8},
                        ],
                    },
                    {  # the store name at the top
                        "geometry": ((0.0, 0.05), (1.0, 0.09)),
                        "words": [{"value": "WALMART", "confidence": 0.9}],
                    },
                    {  # empty line -> dropped
                        "geometry": ((0.0, 0.5), (1.0, 0.5)),
                        "words": [],
                    },
                ]
            }]
        }]
    }
    lines = _lines_from_export(export)
    assert [ln.text for ln in lines] == ["WALMART", "TOTAL 12.00"]   # sorted top->bottom
    assert lines[1].conf == 0.9                                       # mean(1.0, 0.8)
    assert lines[0].conf == 0.9


def test_lines_from_export_defaults_confidence_when_missing():
    export = {"pages": [{"blocks": [{"lines": [
        {"geometry": ((0, 0.1), (1, 0.2)), "words": [{"value": "X"}]},
    ]}]}]}
    assert _lines_from_export(export)[0].conf == 1.0   # no confidence -> trust (1.0)


# --- the OCR confidence arms the arithmetic abstain guard --------------------

def test_low_ocr_confidence_arms_arithmetic_abstain_guard():
    # total (50.00) wildly != subtotal+tax (5.50), but the total line was read at low
    # confidence -> likely a misread, so arithmetic abstains instead of crying fraud.
    lines = [
        _ln("Subtotal 5.00", y=0.80, conf=1.0),
        _ln("Tax 0.50", y=0.85, conf=1.0),
        _ln("Total 50.00", y=0.90, conf=0.3),
    ]
    r = _receipt_from_lines(lines, doc_id="d", image_path="/i.jpg")
    assert r.field_confidence == {"total": 0.3}
    sig = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert sig.abstained


def test_high_ocr_confidence_lets_arithmetic_flag_real_mismatch():
    # Contrast / non-circularity: the same mismatch read at full confidence IS fraud.
    lines = [
        _ln("Subtotal 5.00", y=0.80, conf=1.0),
        _ln("Tax 0.50", y=0.85, conf=1.0),
        _ln("Total 50.00", y=0.90, conf=1.0),
    ]
    r = _receipt_from_lines(lines, doc_id="d", image_path="/i.jpg")
    assert r.field_confidence == {}
    sig = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert not sig.abstained and sig.score > 0.6


# --- extractor contract + registry ------------------------------------------

def test_doctr_handles_only_image_route():
    ex = DocTROCRExtractor()
    assert ex.can_handle(DocumentType.IMAGE)
    assert not ex.can_handle(DocumentType.STRUCTURED)
    assert not ex.can_handle(DocumentType.PDF)


def test_doctr_available_returns_bool_reason_tuple():
    ok, reason = DocTROCRExtractor().available()
    assert isinstance(ok, bool) and isinstance(reason, str)


def test_image_extractors_includes_both_paradigms():
    names = [e.name for e in image_extractors()]
    assert "doctr" in names and any(n != "doctr" for n in names)
    assert all(e.can_handle(DocumentType.IMAGE) for e in image_extractors())


def test_image_extractor_for_spec_maps_each_spec():
    assert isinstance(image_extractor_for_spec("doctr"), DocTROCRExtractor)
    assert isinstance(image_extractor_for_spec("vlm"), QwenVLExtractor)
    # anything else is treated as an HF checkpoint id for the VLM
    custom = image_extractor_for_spec("Qwen/Qwen2.5-VL-7B-Instruct")
    assert isinstance(custom, QwenVLExtractor)
    assert custom.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
