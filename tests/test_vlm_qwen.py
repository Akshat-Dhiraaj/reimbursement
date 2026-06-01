"""Tests for the Qwen2.5-VL extractor's *pure* parts — JSON parsing and dict->Receipt
mapping — with no model load and no heavy deps imported. These are what make the
extractor robust to the messy text a VLM actually returns."""

from __future__ import annotations

from datetime import date as Date

from slipguard.extractors import QwenVLExtractor, image_extractors
from slipguard.extractors.vlm_qwen import (
    _num,
    _parse_date,
    _parse_json_object,
    _to_receipt,
)
from slipguard.models import DocumentType


def test_parse_json_plain():
    assert _parse_json_object('{"total": 5}') == {"total": 5}


def test_parse_json_strips_code_fences():
    text = '```json\n{"vendor_name": "Costco", "total": 9.99}\n```'
    assert _parse_json_object(text) == {"vendor_name": "Costco", "total": 9.99}


def test_parse_json_ignores_surrounding_prose():
    text = 'Sure, here is the receipt:\n{"total": 1.5}\nLet me know if you need more.'
    assert _parse_json_object(text) == {"total": 1.5}


def test_parse_json_returns_none_on_garbage():
    assert _parse_json_object("no json here") is None
    assert _parse_json_object("[1, 2, 3]") is None  # a list is not a Receipt object


def test_num_coercions():
    assert _num("$1,234.50") == 1234.5
    assert _num(12) == 12.0
    assert _num(3.5) == 3.5
    assert _num(None) is None
    assert _num("n/a") is None
    assert _num(True) is None  # bool must not become 1.0


def test_parse_date_formats():
    assert _parse_date("2020-01-02") == Date(2020, 1, 2)
    assert _parse_date("01/02/2020") == Date(2020, 1, 2)
    assert _parse_date("garbage") is None
    assert _parse_date(None) is None


def test_to_receipt_maps_fields_and_skips_amountless_items():
    data = {
        "vendor_name": "Trader Joe's",
        "date": "2021-03-04",
        "currency": "USD",
        "subtotal": 10.0,
        "tax_amount": "$0.90",
        "total": 10.9,
        "line_items": [
            {"description": "Milk", "quantity": 1, "unit_price": 3.0, "amount": 3.0},
            {"description": "no price", "quantity": 1},  # no amount -> skipped
        ],
    }
    r = _to_receipt(data, doc_id="img1", image_path="/x/img1.jpg")
    assert r.vendor_name == "Trader Joe's"
    assert r.date == Date(2021, 3, 4)
    assert r.subtotal == 10.0
    assert r.tax_amount == 0.90
    assert r.total == 10.9
    assert len(r.line_items) == 1
    assert r.line_items[0].amount == 3.0
    assert r.source is DocumentType.IMAGE
    assert r.image_path == "/x/img1.jpg" and r.source_path == "/x/img1.jpg"


def test_to_receipt_handles_missing_fields():
    r = _to_receipt({}, doc_id="img2", image_path="/x/img2.jpg")
    assert r.vendor_name == "(unknown)"
    assert r.date is None
    assert r.subtotal is None and r.tax_amount is None and r.total is None
    assert r.line_items == []
    assert r.doc_id == "img2"


def test_unit_price_defaults_to_amount_when_missing():
    data = {"line_items": [{"description": "X", "amount": 7.5}]}
    r = _to_receipt(data, doc_id="i", image_path="/i.jpg")
    assert r.line_items[0].unit_price == 7.5
    assert r.line_items[0].quantity == 1.0  # default qty


def test_extractor_handles_only_image_route():
    ex = QwenVLExtractor()
    assert ex.can_handle(DocumentType.IMAGE)
    assert not ex.can_handle(DocumentType.STRUCTURED)
    assert not ex.can_handle(DocumentType.PDF)


def test_available_returns_bool_reason_tuple():
    ok, reason = QwenVLExtractor().available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


def test_image_extractors_registry_and_model_override():
    assert all(e.can_handle(DocumentType.IMAGE) for e in image_extractors())
    custom = image_extractors("Qwen/Qwen2.5-VL-7B-Instruct")
    assert custom[0].model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
