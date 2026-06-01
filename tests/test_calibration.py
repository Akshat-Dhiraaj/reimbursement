"""Tests for the confidence calibration study (eval/calibration.py).

Dependency-free: the analysis (``summarize_calibration`` and its helpers) is pure and
tested on hand-built rows, and a stub Extractor stands in for the heavy VLM so the
collection loop is exercised without a model. The headline assertion is the honest one:
when correct and incorrect reads carry the *same* confidence (the "confident misread"
case the audit found), AUC is 0.5 — the signal is uninformative on its own.
"""

from __future__ import annotations

from datetime import date as Date

import pytest

from slipguard.eval.calibration import (
    ConfRow,
    _confidence_of,
    collect_confidence_rows,
    summarize_calibration,
)
from slipguard.extractors.base import Extractor
from slipguard.models import DocumentType, LineItem, Receipt


# --- _confidence_of: absent == fully trusted (1.0) -----------------------------

def test_confidence_of_defaults_to_one_when_absent():
    r = Receipt(doc_id="d", vendor_name="V", date=Date(2020, 1, 1),
                field_confidence={"total": 0.4})
    assert _confidence_of(r, "total") == 0.4
    assert _confidence_of(r, "subtotal") == 1.0  # not recorded -> read at full confidence


# --- summarize_calibration: separation (AUC) -----------------------------------

def test_auc_is_one_when_confidence_perfectly_separates():
    rows = [
        ConfRow("total", 0.95, True), ConfRow("total", 1.0, True),
        ConfRow("total", 0.60, False), ConfRow("total", 0.70, False),
    ]
    allf = summarize_calibration(rows).per_field[0]
    assert allf.name == "ALL"
    assert allf.auc == 1.0  # every correct read scored above every incorrect one


def test_auc_is_half_when_misreads_are_as_confident_as_correct_reads():
    # The honest verdict in miniature: confidence cannot tell the two apart.
    rows = [
        ConfRow("total", 0.8, True), ConfRow("total", 0.8, True),
        ConfRow("total", 0.8, False), ConfRow("total", 0.8, False),
    ]
    allf = summarize_calibration(rows).per_field[0]
    assert allf.auc == 0.5


def test_field_counts_accuracy_and_mean_confidence():
    rows = [
        ConfRow("total", 0.9, True),
        ConfRow("total", 0.6, False),
        ConfRow("total", 0.7, False),
    ]
    allf = summarize_calibration(rows).per_field[0]
    assert allf.n == 3 and allf.n_incorrect == 2
    assert allf.accuracy == pytest.approx(1 / 3)
    assert allf.mean_conf_correct == pytest.approx(0.9)
    assert allf.mean_conf_incorrect == pytest.approx((0.6 + 0.7) / 2)


def test_per_field_breakdown_has_all_plus_each_present_money_field():
    rows = [
        ConfRow("total", 0.9, True),
        ConfRow("subtotal", 0.8, False),
    ]
    names = [f.name for f in summarize_calibration(rows).per_field]
    assert names[0] == "ALL"
    assert set(names[1:]) == {"subtotal", "total"}  # tax_amount had no rows -> omitted


# --- summarize_calibration: reliability bins -----------------------------------

def test_reliability_bins_place_reads_and_isolate_full_confidence():
    rows = [
        ConfRow("total", 0.55, False),  # -> [0.0, 0.6)
        ConfRow("total", 0.95, True),   # -> [0.9, 1.0)
        ConfRow("total", 1.0, True),    # -> ==1.0 (its own row)
        ConfRow("total", 1.0, True),
    ]
    bins = {(b.lo, b.hi): b for b in summarize_calibration(rows).bins}
    assert bins[(0.0, 0.6)].n == 1 and bins[(0.0, 0.6)].accuracy == 0.0
    assert bins[(0.9, 1.0)].n == 1 and bins[(0.9, 1.0)].accuracy == 1.0
    full = bins[(1.0, 1.0)]
    assert full.n == 2 and full.accuracy == 1.0  # the exact-1.0 mass is not smeared down


# --- summarize_calibration: abstain threshold sweep ----------------------------

def test_threshold_sweep_counts_misreads_caught_and_correct_dropped():
    rows = [
        ConfRow("total", 0.60, False),  # caught at T>=0.7
        ConfRow("total", 0.80, False),  # only caught at T>=0.9
        ConfRow("total", 0.65, True),   # wrongly dropped at T>=0.7 (the cost)
        ConfRow("total", 1.00, True),   # never dropped
    ]
    sweep = {s.threshold: s for s in summarize_calibration(rows).sweep}
    s70 = sweep[0.7]
    assert (s70.misreads_caught, s70.total_incorrect) == (1, 2)
    assert s70.recall == pytest.approx(0.5)
    assert (s70.correct_abstained, s70.total_correct) == (1, 2)
    assert s70.cost == pytest.approx(0.5)


def test_report_str_has_the_three_views():
    rows = [ConfRow("total", 0.9, True), ConfRow("total", 0.6, False)]
    text = str(summarize_calibration(rows, "qwen2-vl"))
    assert "qwen2-vl" in text
    assert "AUC" in text and "Reliability" in text and "Abstain sweep" in text


# --- collect_confidence_rows: pairing oracle truth with extractor confidence ----

def _truth(doc_id="t1", subtotal=10.0, tax=0.9, total=10.9):
    return Receipt(doc_id=doc_id, vendor_name="V", date=Date(2020, 1, 1),
                   subtotal=subtotal, tax_amount=tax, total=total,
                   line_items=[LineItem("i", 1, 1.0, 1.0)],
                   image_path=f"/img/{doc_id}.jpg")


class _StubExtractor(Extractor):
    """Returns a preset (value, confidence) read per doc, so we can drive the loop."""
    name = "stub"
    handles = (DocumentType.IMAGE,)

    def __init__(self, by_id):
        self._by_id = by_id

    def extract(self, path, doc_id=None):
        return self._by_id[doc_id]


def test_collect_records_money_field_only_when_both_sides_present():
    truth = _truth("a")  # oracle has subtotal/tax/total
    # Model reads total wrong, omits tax_amount, reads subtotal right but low-confidence.
    pred = Receipt(doc_id="a", vendor_name="V", date=Date(2020, 1, 1),
                   subtotal=10.0, tax_amount=None, total=99.0,
                   field_confidence={"subtotal": 0.3, "total": 0.8})
    rows = collect_confidence_rows(_StubExtractor({"a": pred}), [truth])
    by = {r.name: r for r in rows}
    assert set(by) == {"subtotal", "total"}        # tax omitted by model -> no row
    assert by["subtotal"].confidence == 0.3 and by["subtotal"].correct is True
    assert by["total"].confidence == 0.8 and by["total"].correct is False


def test_collect_skips_field_the_oracle_lacks():
    truth = _truth("a", subtotal=None)  # oracle has no subtotal to judge against
    pred = Receipt(doc_id="a", vendor_name="V", date=Date(2020, 1, 1),
                   subtotal=10.0, tax_amount=0.9, total=10.9)
    rows = collect_confidence_rows(_StubExtractor({"a": pred}), [truth])
    assert "subtotal" not in {r.name for r in rows}


def test_collect_treats_extractor_error_as_no_rows_not_crash():
    class _Broken(Extractor):
        name = "broken"
        handles = (DocumentType.IMAGE,)

        def extract(self, path, doc_id=None):
            raise RuntimeError("cannot read")

    assert collect_confidence_rows(_Broken(), [_truth("a")]) == []
