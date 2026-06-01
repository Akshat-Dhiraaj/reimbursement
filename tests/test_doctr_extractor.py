"""Tests for the docTR OCR+KIE extractor's *pure* parts — date finding, money
parsing, the OCR-export flattener, and the keyword/position KIE that maps read lines
onto a Receipt — with no model load and no doctr/torch imported. The heuristics are
the whole point of this extractor (a transparent rival to the VLM on the same oracle),
so they are unit-tested directly; ``extract()`` itself only loads the OCR model and
delegates to these functions.

The final block proves the genuine OCR recognition confidence is wired through to the
arithmetic abstain guard, exactly as the VLM's parse-completeness confidence is.
"""

from __future__ import annotations

from datetime import date as Date

from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.extractors import (
    DocTROCRExtractor,
    QwenVLExtractor,
    image_extractor_for_spec,
    image_extractors,
)
from slipguard.extractors.doctr_ocr import (
    _Line,
    _find_date,
    _geom_x_min,
    _geom_y_center,
    _last_money,
    _lines_from_export,
    _merge_rows,
    _mk_date,
    _pick_line_items,
    _pick_money_field,
    _pick_vendor,
    _receipt_from_lines,
)
from slipguard.models import DocumentType


def _ln(text: str, y: float = 0.5, conf: float = 1.0, x: float = 0.0) -> _Line:
    return _Line(text=text, y=y, conf=conf, x=x)


# --- date finding ------------------------------------------------------------

def test_mk_date_expands_two_digit_year():
    assert _mk_date(20, 1, 2) == Date(2020, 1, 2)   # 00-69 -> 20xx
    assert _mk_date(95, 1, 2) == Date(1995, 1, 2)   # 90-99 -> 19xx, above the 1990 floor


def test_mk_date_rejects_out_of_range_and_invalid():
    assert _mk_date(1850, 1, 1) is None             # before 1990 -> not a real date
    assert _mk_date(85, 1, 1) is None               # '85 -> 1985, below the 1990 floor
    assert _mk_date(2020, 13, 1) is None            # month 13
    assert _mk_date(2020, 2, 30) is None            # Feb 30


def test_find_date_iso_ymd():
    assert _find_date("Date: 2020-01-02") == Date(2020, 1, 2)


def test_find_date_numeric_is_month_first_then_day_first():
    assert _find_date("01/02/2020") == Date(2020, 1, 2)   # MDY read first
    assert _find_date("13/02/2020") == Date(2020, 2, 13)  # MDY impossible -> DMY


def test_find_date_textual_month_both_orders():
    assert _find_date("2 Jan 2020") == Date(2020, 1, 2)
    assert _find_date("January 2, 2020") == Date(2020, 1, 2)


def test_find_date_none_when_absent():
    assert _find_date("no date here, just text") is None


# --- money on a line ---------------------------------------------------------

def test_last_money_takes_rightmost():
    # description carries an incidental number; the amount sits in the right column
    assert _last_money("2 x Coffee 3.50  7.00") == (7.0, "7.00")


def test_last_money_price_shape_screens_non_prices():
    # require_price_shape skips the phone-like token and returns the real price
    assert _last_money("Call 5551234 total 3.50", require_price_shape=True) == (3.5, "3.50")
    # a lone integer quantity has no 2-decimal price shape -> nothing
    assert _last_money("Qty 2", require_price_shape=True) == (None, None)


def test_last_money_none_when_no_number():
    assert _last_money("Thank you for shopping") == (None, None)


# --- OCR export -> lines -----------------------------------------------------

def test_geom_y_center_box_and_polygon():
    assert _geom_y_center(((0.0, 0.1), (1.0, 0.3))) == 0.2          # straight box
    assert _geom_y_center([(0, 0.1), (1, 0.1), (1, 0.3), (0, 0.3)]) == 0.2  # 4-pt polygon
    assert _geom_y_center("bogus") == 0.0                           # defensive -> top


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


# --- row reconstruction (docTR splits a row's label & amount apart) ----------

def test_geom_x_min_box_and_polygon():
    assert _geom_x_min(((0.6, 0.1), (0.9, 0.3))) == 0.6           # straight box left edge
    assert _geom_x_min([(0.6, 0.1), (0.9, 0.1), (0.9, 0.3), (0.6, 0.3)]) == 0.6  # polygon
    assert _geom_x_min("bogus") == 0.0                            # defensive -> left


def test_merge_rows_joins_same_height_lines_x_ordered():
    # label in the left column, amount in the right column, same height -> one row,
    # text ordered left-to-right by x.
    rows = _merge_rows([_ln("55.96", y=0.80, x=0.75), _ln("SUBTOTAL", y=0.81, x=0.05)])
    assert len(rows) == 1
    assert rows[0].text == "SUBTOTAL 55.96"


def test_merge_rows_keeps_distinct_rows_separate():
    # rows further apart than y_tol are not merged.
    rows = _merge_rows([_ln("SUBTOTAL 5.00", y=0.80), _ln("TOTAL 5.50", y=0.90)])
    assert [r.text for r in rows] == ["SUBTOTAL 5.00", "TOTAL 5.50"]


def test_merge_rows_keeps_amount_as_rightmost_money_token():
    # the real 'TAX1' failure: the label line carries a stray digit. x-ordering must put
    # the right-column amount last so _last_money reads 4.48, not the '1' in 'TAX1'.
    rows = _merge_rows([_ln("4.48", y=0.85, x=0.78), _ln("TAX1", y=0.85, x=0.05)])
    assert rows[0].text == "TAX1 4.48"
    assert _last_money(rows[0].text)[0] == 4.48


# --- KIE: lines -> fields ----------------------------------------------------

def test_pick_vendor_most_letter_heavy_in_top_band():
    lines = [
        _ln("12/02/2020", y=0.02),
        _ln("BIG BAZAAR STORE", y=0.06),
        _ln("Milk 3.50", y=0.6),
    ]
    assert _pick_vendor(lines) == "BIG BAZAAR STORE"


def test_pick_money_field_takes_bottommost_keyword_line():
    lines = [
        _ln("Total items 3", y=0.70),       # no money -> ignored
        _ln("TOTAL 10.00", y=0.80),
        _ln("TOTAL DUE 12.00", y=0.92),      # bottom-most -> the final figure
    ]
    val = _pick_money_field(lines, ("total", "total due"))
    assert val is not None and val[0] == 12.0


def test_pick_total_excludes_subtotal_line():
    # "subtotal" contains "total", so total-matching must exclude subtotal rows.
    lines = [
        _ln("Subtotal 10.00", y=0.80, conf=0.95),
        _ln("Total 11.00", y=0.90, conf=0.90),
    ]
    sub = _pick_money_field(lines, ("subtotal", "sub total"))
    tot = _pick_money_field(lines, ("total",), exclude_kw=("subtotal", "sub total"))
    assert sub == (10.0, 0.95)
    assert tot == (11.0, 0.90)


def test_pick_line_items_skips_summary_and_is_internally_consistent():
    lines = [
        _ln("Milk 3.50"),
        _ln("Bread 2.00"),
        _ln("Subtotal 5.50"),       # summary keyword -> not an item
        _ln("Phone 5551234"),       # no price shape -> not an item
        _ln("Thanks!"),             # no money -> not an item
    ]
    items, confs = _pick_line_items(lines)
    assert [it.description for it in items] == ["Milk", "Bread"]
    assert len(confs) == 2
    # qty=1, unit_price=amount -> each item reconciles, so arithmetic can only ever
    # flag a subtotal-vs-sum gap (the real signal), never our own harvested rows.
    for it in items:
        assert it.quantity == 1.0 and it.unit_price == it.amount


# --- lines -> Receipt --------------------------------------------------------

def test_receipt_from_lines_maps_all_fields():
    lines = [
        _ln("COSTCO WHOLESALE", y=0.05),
        _ln("01/02/2020", y=0.10),
        _ln("Milk 3.00", y=0.40),
        _ln("Eggs 2.00", y=0.45),
        _ln("Subtotal 5.00", y=0.80),
        _ln("Tax 0.50", y=0.85),
        _ln("Total 5.50", y=0.90),
    ]
    r = _receipt_from_lines(lines, doc_id="d1", image_path="/img/d1.jpg")
    assert r.vendor_name == "COSTCO WHOLESALE"
    assert r.date == Date(2020, 1, 2)
    assert r.subtotal == 5.0 and r.tax_amount == 0.5 and r.total == 5.5
    assert len(r.line_items) == 2
    assert r.source is DocumentType.IMAGE
    assert r.image_path == "/img/d1.jpg" and r.source_path == "/img/d1.jpg"


def test_receipt_from_lines_recovers_split_two_column_summary():
    # The real WildReceipt failure: docTR emits each summary row's amount and label as
    # SEPARATE lines at the same height (amount in the right column, x larger). The
    # row-merge must reunite them so subtotal/tax/total are read — and the orphaned
    # amounts must NOT survive as fake line items.
    lines = [
        _ln("KOREAN RESTAURANT", y=0.08, x=0.10),
        _ln("DATE 12/30/2016", y=0.34, x=0.10),
        _ln("13.99", y=0.45, x=0.80), _ln("BIBIM OCTOPU", y=0.45, x=0.05),
        _ln("55.96", y=0.78, x=0.80), _ln("SUBTOTAL", y=0.79, x=0.05),
        _ln("4.48", y=0.85, x=0.80), _ln("TAX1", y=0.86, x=0.05),
        _ln("60.44", y=0.93, x=0.80), _ln("TOTAL", y=0.94, x=0.05),
    ]
    r = _receipt_from_lines(lines, doc_id="d", image_path="/i.jpg")
    assert r.subtotal == 55.96 and r.tax_amount == 4.48 and r.total == 60.44
    assert len(r.line_items) == 1                       # only the real item, no orphans
    assert "BIBIM" in r.line_items[0].description
    assert r.date == Date(2016, 12, 30)


def test_receipt_from_lines_clean_read_records_no_confidence():
    # Every line read at conf 1.0 -> field_confidence empty == fully trusted, so the
    # arithmetic guard behaves exactly as on the oracle path (no regression).
    lines = [
        _ln("Subtotal 5.00", y=0.80),
        _ln("Tax 0.50", y=0.85),
        _ln("Total 5.50", y=0.90),
    ]
    r = _receipt_from_lines(lines, doc_id="d", image_path="/i.jpg")
    assert r.field_confidence == {}


def test_receipt_from_lines_records_low_recognition_confidence():
    lines = [
        _ln("Subtotal 5.00", y=0.80, conf=1.0),
        _ln("Total 5.50", y=0.90, conf=0.3),   # garbled read of the total line
    ]
    r = _receipt_from_lines(lines, doc_id="d", image_path="/i.jpg")
    assert r.field_confidence == {"total": 0.3}   # only the sub-1.0 read is recorded


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
