"""Tests for the CORD oracle loader (data/cord.py).

Like the WildReceipt tests, these exercise the **pure** gt_parse -> Receipt mapping on
hand-built dicts — no network / `datasets` download needed. The cases pin the real CORD
quirks the FP audit surfaced: Indonesian grouping, dict-or-list menus, nested add-on rows,
line-level discounts, unitprice*cnt fallback, and the no-vendor/no-date scope.
"""

from __future__ import annotations

from slipguard.data.cord import _line_amount, _menu_lines, _qty, _receipt_from_gt
from slipguard.models import DocumentType


# --- pure helpers ------------------------------------------------------------

def test_qty_reads_first_number_default_one():
    assert _qty("2") == 2.0
    assert _qty("x1") == 1.0
    assert _qty("1X") == 1.0
    assert _qty(None) == 1.0
    assert _qty("") == 1.0


def test_line_amount_prefers_price_then_unitprice_times_count():
    assert _line_amount({"price": "24,000"}) == 24000.0          # grouping -> whole IDR
    assert _line_amount({"unitprice": "4.000", "cnt": "2"}) == 8000.0  # no price -> unit*cnt
    assert _line_amount({"nm": "modifier only"}) is None         # nothing to read


def test_line_amount_nets_a_labelled_discount():
    # A discounted duplicate line: price 85,500 with discountprice -85,500 nets to 0, so it
    # does not double-count against the subtotal (this was a real subtotal!=sum FP cause).
    assert _line_amount({"price": "85,500", "discountprice": "-85,500"}) == 0.0


# --- menu flattening ---------------------------------------------------------

def test_menu_accepts_single_dict_and_flattens_subitems():
    # The 28,000 receipt: a parent line plus a nested add-on that counts toward the subtotal.
    menu = {"nm": "JASMINE", "cnt": "1", "price": "24,000",
            "sub": {"nm": "COCONUT", "price": "4,000"}}
    items = _menu_lines(menu)
    assert [li.amount for li in items] == [24000.0, 4000.0]
    assert sum(li.amount for li in items) == 28000.0  # == subtotal_price 28,000


def test_menu_accepts_list_and_skips_priceless_rows():
    menu = [{"nm": "A", "price": "17500"}, {"nm": "B", "price": "46,000"},
            {"nm": "C (modifier)"}]  # no price -> skipped
    assert [li.amount for li in _menu_lines(menu)] == [17500.0, 46000.0]


# --- full oracle mapping -----------------------------------------------------

def test_receipt_from_gt_maps_money_and_locale():
    gt = {
        "menu": [{"nm": "X", "price": "17500"}, {"nm": "Y", "price": "46,000"}],
        "sub_total": {"subtotal_price": "63.500", "tax_price": "0"},
        "total": {"total_price": "63500"},
    }
    r = _receipt_from_gt(gt, "cord-test:0", image_path="img.png")
    assert [li.amount for li in r.line_items] == [17500.0, 46000.0]
    assert r.subtotal == 63500.0 and r.total == 63500.0
    assert r.currency == "IDR" and r.country == "ID"   # so the GSTIN detector abstains
    assert r.vendor_name == "(unknown)" and r.date is None  # CORD labels carry neither
    assert r.source is DocumentType.IMAGE and r.image_path == "img.png"


def test_receipt_from_gt_maps_service_charge_and_discount():
    # CORD's sub_total carries service_price / discount_price (×12 / ×6 in CORD-test); the
    # richer Receipt maps them so `total` reconciles as subtotal + tax + service - discount
    # (this is what cut the CORD clean-oracle FP 0.170 -> 0.030).
    gt = {
        "menu": [{"nm": "X", "price": "100000"}],
        "sub_total": {"subtotal_price": "100000", "tax_price": "10000",
                      "service_price": "5000", "discount_price": "3000"},
        "total": {"total_price": "112000"},  # 100000 + 10000 + 5000 - 3000
    }
    r = _receipt_from_gt(gt, "cord-test:svc")
    assert r.service_charge == 5000.0 and r.discount == 3000.0
    from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
    assert ArithmeticConsistencyDetector().score(r).score < 0.1   # reconciles -> no FP


def test_receipt_from_gt_tolerates_empty_or_malformed_blocks():
    # sub_total/total absent or not dicts must not crash; money fields just read None.
    r = _receipt_from_gt({"menu": []}, "cord-test:1")
    assert r.subtotal is None and r.tax_amount is None and r.total is None
    assert r.line_items == []
    r2 = _receipt_from_gt({"sub_total": "junk", "total": None}, "cord-test:2")
    assert r2.subtotal is None and r2.total is None
