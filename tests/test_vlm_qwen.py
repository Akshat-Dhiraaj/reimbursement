"""Tests for the Qwen-VL extractor's *pure* parts — JSON parsing, dict->Receipt
mapping, and the parse-completeness + token-logprob confidence it records — with no model
load and no heavy deps imported. These are what make the extractor robust to the messy text
a VLM actually returns, and what arms the arithmetic abstain guard on under-capture and on
low-confidence scalar misreads."""

from __future__ import annotations

from datetime import date as Date

from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.extractors import QwenVLExtractor, image_extractors
from slipguard.extractors.vlm_qwen import (
    _field_confidence_from_tokens,
    _incremental_spans,
    _num,
    _parse_date,
    _parse_json_object,
    _to_receipt,
)
from slipguard.models import DocumentType


def _decoder(pieces: list[str]):
    """A trivial, GPU-free stand-in for a tokenizer's ``decode``: token id ``i`` is the
    i-th string piece, so ``decode(ids)`` concatenates them. Lets the incremental-span and
    logprob helpers be tested with hand-built token boundaries and per-token probabilities."""
    return lambda ids: "".join(pieces[i] for i in ids)


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


# ---- Parse-completeness confidence ----------------------------------------

def test_field_confidence_empty_when_all_items_parse():
    # A clean extraction records nothing -> empty field_confidence == fully trusted,
    # so the guard behaves exactly as before (no regression on good extractions).
    data = {
        "line_items": [
            {"description": "A", "quantity": 1, "unit_price": 2.0, "amount": 2.0},
            {"description": "B", "quantity": 1, "unit_price": 3.0, "amount": 3.0},
        ],
        "subtotal": 5.0, "total": 5.0,
    }
    r = _to_receipt(data, doc_id="i", image_path="/i.jpg")
    assert r.field_confidence == {}


def test_field_confidence_absent_when_no_line_items():
    r = _to_receipt({"subtotal": 5.0, "total": 5.0}, doc_id="i", image_path="/i.jpg")
    assert r.field_confidence == {}


def test_field_confidence_records_line_item_parse_ratio():
    # Model emits 4 line items but only 1 carries a parseable amount -> ratio 0.25,
    # counting non-dict junk and amountless items as emitted-but-unusable.
    data = {"line_items": [
        {"description": "ok", "amount": 2.0},
        {"description": "no amount"},
        {"description": "bad amount", "amount": "n/a"},
        "totally malformed",
    ]}
    r = _to_receipt(data, doc_id="i", image_path="/i.jpg")
    assert len(r.line_items) == 1
    assert r.field_confidence == {"line_items": 0.25}


# ---- The confidence actually arms the (previously dormant) arithmetic guard ----

def test_low_parse_ratio_arms_arithmetic_abstain_guard():
    # End-to-end: 5 items emitted, 1 parsed -> line_items confidence 0.2. The
    # subtotal!=sum(items) gap is a capture artifact, so arithmetic abstains.
    data = {
        "line_items": [
            {"description": "kept", "amount": 2.0},
            {"description": "drop1"}, {"description": "drop2"},
            {"description": "drop3"}, {"description": "drop4"},
        ],
        "subtotal": 100.0, "total": 100.0,
    }
    r = _to_receipt(data, doc_id="i", image_path="/i.jpg")
    assert r.field_confidence["line_items"] == 0.2
    sig = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert sig.abstained  # low-confidence extraction -> abstain, not "fraud"


def test_clean_extraction_lets_arithmetic_flag_real_mismatch():
    # Contrast / non-circularity: with every emitted item parsed (confidence empty ==
    # trusted), the same subtotal!=sum(items) gap IS reported as fraud. The guard mutes
    # only on low-confidence extraction, it is not a blanket suppressor.
    data = {
        "line_items": [{"description": "only", "amount": 2.0}],
        "subtotal": 100.0, "total": 100.0,
    }
    r = _to_receipt(data, doc_id="i", image_path="/i.jpg")
    assert r.field_confidence == {}
    sig = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert not sig.abstained and sig.score > 0.6


# ---- Incremental token->char span mapping -----------------------------------

def test_incremental_spans_maps_each_token_to_its_chars():
    # Each span must slice the decoded text back to exactly the token that produced it —
    # this is the alignment the logprob->field step relies on.
    pieces = ['{"total": ', "58", ".", "22", "}"]
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    assert text == '{"total": 58.22}'
    assert [text[s:e] for s, e in spans] == pieces


def test_incremental_spans_handles_zero_width_token():
    # A token that decodes to nothing (e.g. a skipped special token mid-sequence) gets an
    # empty span and so covers no field value, rather than shifting later spans.
    pieces = ["58", "", "22"]
    text, spans = _incremental_spans(_decoder(pieces), [0, 1, 2])
    assert text == "5822"
    assert spans[1][0] == spans[1][1]  # zero-width
    assert [text[s:e] for s, e in spans] == ["58", "", "22"]


# ---- Token-logprob confidence on the scalar money fields --------------------

def test_field_confidence_records_min_token_prob_over_value_digits():
    # The value 58.22 spans three tokens (58 | . | 22); the least-confident digit (the
    # 0.40 on '22') sets the field's confidence — min, so one shaky digit drags it down.
    pieces = ['{"total": ', "58", ".", "22", "}"]
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    probs = [0.99, 0.95, 0.99, 0.40, 0.99]
    assert _field_confidence_from_tokens(text, spans, probs) == {"total": 0.4}


def test_field_confidence_skips_null_and_absent_fields():
    # A field emitted as null has no digits to score; a field never emitted is skipped too.
    # Only the asserted subtotal (with its shaky 0.30) is recorded.
    pieces = ['{"subtotal": ', "10", ', "tax_amount": null}']
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    probs = [0.99, 0.30, 0.99]
    assert _field_confidence_from_tokens(text, spans, probs) == {"subtotal": 0.3}


def test_field_confidence_records_only_uncertainty():
    # A confident read (rounds to 1.0) records nothing -> field_confidence stays empty
    # (== trusted), mirroring the parse-completeness convention so behaviour is unchanged.
    pieces = ['{"total": ', "11", "}"]
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    assert _field_confidence_from_tokens(text, spans, [0.99, 0.9997, 0.99]) == {}


def test_field_confidence_handles_quoted_number():
    # The prompt forbids quotes, but if the model emits "total": "58" we still score the
    # digits (the optional leading quote is outside the captured value).
    pieces = ['{"total": "', "58", '"}']
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    assert _field_confidence_from_tokens(text, spans, [0.99, 0.55, 0.99]) == {"total": 0.55}


# ---- The confidence actually arms the (previously dormant) arithmetic guard --

def test_low_token_prob_arms_arithmetic_abstain_guard():
    # End-to-end: a receipt whose total does not reconcile (would flag as fraud), but the
    # model emitted the total's digits with low probability -> low scalar confidence ->
    # arithmetic abstains (a misread, not fraud). The scalar analogue of the under-capture
    # guard, catching exactly the confident-misread gap parse-completeness cannot see.
    r = _to_receipt(
        {"subtotal": 10.0, "tax_amount": 0.0, "total": 999.0,
         "line_items": [{"description": "x", "amount": 10.0}]},
        doc_id="i", image_path="/i.jpg")
    pieces = ['{"total": ', "999", "}"]
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    r.field_confidence.update(_field_confidence_from_tokens(text, spans, [0.99, 0.25, 0.99]))
    assert r.field_confidence["total"] == 0.25
    sig = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert sig.abstained


def test_confident_misread_still_flags_documenting_the_honest_limit():
    # The honest limit, made a test: a logprob measures self-assurance, not truth. A total
    # the model emitted *confidently* (high token prob) but wrongly is trusted, so
    # arithmetic still flags the 999 != 10 mismatch. Token logprobs catch hesitant misreads,
    # not confidently-wrong ones — and the guard is not a blanket mute.
    r = _to_receipt(
        {"subtotal": 10.0, "tax_amount": 0.0, "total": 999.0,
         "line_items": [{"description": "x", "amount": 10.0}]},
        doc_id="i", image_path="/i.jpg")
    pieces = ['{"total": ', "999", "}"]
    text, spans = _incremental_spans(_decoder(pieces), list(range(len(pieces))))
    r.field_confidence.update(_field_confidence_from_tokens(text, spans, [0.99, 0.98, 0.99]))
    assert r.field_confidence["total"] >= 0.5  # confident (even if wrong) -> guard trusts it
    sig = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert not sig.abstained and sig.score > 0.6
