import json
from datetime import date

from slipguard.data.wildreceipt import (
    _parse_date,
    _record_to_receipt,
    load_receipts,
)
from slipguard.models import DocumentType


def _ann(label: int, text: str) -> dict:
    return {"label": label, "text": text}


# --- pure helpers ------------------------------------------------------------
# (money parsing is covered in test_money.py — the oracle delegates to money.parse_money)

def test_parse_date_accepts_common_formats():
    assert _parse_date("01/15/2019") == date(2019, 1, 15)
    assert _parse_date("2019-01-15") == date(2019, 1, 15)
    assert _parse_date("Jan 15, 2019") == date(2019, 1, 15)
    assert _parse_date("15 Jan 2019") == date(2019, 1, 15)


def test_parse_date_none_on_unparseable():
    assert _parse_date("not a date") is None
    assert _parse_date("") is None
    assert _parse_date(None) is None


# --- oracle extraction (annotations -> Receipt) ------------------------------

def test_record_to_receipt_maps_fields():
    rec = {
        "file_name": "image_files/x.jpeg",
        "annotations": [
            _ann(1, "COSTCO"),
            _ann(7, "01/15/2019"),
            _ann(11, "Milk"), _ann(15, "3.50"),
            _ann(11, "Bread"), _ann(15, "2.00"),
            _ann(17, "5.50"),
            _ann(19, "0.45"),
            _ann(23, "5.95"),
        ],
    }
    r = _record_to_receipt(rec, "test.txt:0")
    assert r.vendor_name == "COSTCO"
    assert r.date == date(2019, 1, 15)
    assert [li.amount for li in r.line_items] == [3.5, 2.0]
    assert r.subtotal == 5.5 and r.tax_amount == 0.45 and r.total == 5.95
    assert r.source is DocumentType.IMAGE
    assert r.country == "US"           # so the GSTIN/VAT detector abstains, as it should
    assert r.tax_rate is None          # left None so arithmetic skips the rate check


def test_record_to_receipt_parses_european_decimals():
    # Regression: this is the real test.txt:1 (ILIO'S) row. The comma is the decimal
    # point; the old comma-stripping parser turned 129,75 into 12975 (a 100x error that
    # corrupted both the extraction benchmark and the FP audit). The oracle must agree
    # with what the VLM actually reads off the image.
    rec = {
        "file_name": "image_files/x.jpeg",
        "annotations": [
            _ann(1, "ILIO'S"), _ann(1, "Restaurant"),
            _ann(17, "Eur129,75"), _ann(19, "24,65"), _ann(23, "Eur154,40"),
        ],
    }
    r = _record_to_receipt(rec, "test.txt:1")
    assert r.subtotal == 129.75
    assert r.tax_amount == 24.65
    assert r.total == 154.40


def test_record_to_receipt_tolerates_missing_fields():
    rec = {"annotations": [_ann(15, "9.99")]}  # only a price, nothing else
    r = _record_to_receipt(rec, "d")
    assert r.vendor_name == "(unknown)"
    assert r.date is None
    assert r.subtotal is None and r.total is None
    assert len(r.line_items) == 1 and r.line_items[0].amount == 9.99


def test_load_receipts_reads_jsonl(tmp_path):
    rec = {"file_name": "f.jpg", "annotations": [_ann(1, "Shop"), _ann(23, "10.00")]}
    (tmp_path / "test.txt").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    receipts = load_receipts(tmp_path, split="test")
    assert len(receipts) == 1
    assert receipts[0].vendor_name == "Shop" and receipts[0].total == 10.0
