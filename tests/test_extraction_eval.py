"""Tests for the extraction-accuracy benchmark (eval/extraction.py).

Dependency-free: stub Extractors stand in for the OCR/VLM ones (which carry heavy
deps and land in Phase 2). A perfect extractor must score 1.0, a broken one 0.0,
and a field is scored only when the oracle actually has a value for it."""

from __future__ import annotations

from datetime import date as Date

import pytest

from slipguard.eval.extraction import (
    ExtractionReport,
    _money_ok,
    _vendor_ok,
    evaluate_extractor,
    evaluate_extractors,
)
from slipguard.extractors.base import Extractor
from slipguard.models import DocumentType, LineItem, Receipt


def _truth(
    doc_id="t1",
    vendor="Trader Joes",
    date=Date(2020, 1, 2),
    subtotal=10.0,
    tax=0.9,
    total=10.9,
    n_items=2,
):
    items = [LineItem(f"i{i}", 1, 1.0, 1.0) for i in range(n_items)]
    return Receipt(
        doc_id=doc_id, vendor_name=vendor, date=date, subtotal=subtotal,
        tax_amount=tax, total=total, line_items=items, image_path=f"/img/{doc_id}.jpg",
    )


class _PerfectExtractor(Extractor):
    """Echoes the oracle Receipt back — the upper bound, should score 1.0."""
    name = "perfect"
    handles = (DocumentType.IMAGE,)

    def __init__(self, truths):
        self._by_id = {r.doc_id: r for r in truths}

    def extract(self, path, doc_id=None):
        return self._by_id[doc_id]


class _BrokenExtractor(Extractor):
    """Always raises — proves a failing extractor counts as misses, not a crash."""
    name = "broken"
    handles = (DocumentType.IMAGE,)

    def extract(self, path, doc_id=None):
        raise RuntimeError("cannot read this document")


class _PerturbedExtractor(Extractor):
    """Echoes the truth but shifts the total and optionally breaks the vendor."""
    name = "perturbed"
    handles = (DocumentType.IMAGE,)

    def __init__(self, truths, total_delta=0.0, wrong_vendor=False):
        self._by_id = {r.doc_id: r for r in truths}
        self.total_delta = total_delta
        self.wrong_vendor = wrong_vendor

    def extract(self, path, doc_id=None):
        t = self._by_id[doc_id]
        return Receipt(
            doc_id=t.doc_id,
            vendor_name="Completely Different Co" if self.wrong_vendor else t.vendor_name,
            date=t.date, subtotal=t.subtotal, tax_amount=t.tax_amount,
            total=(t.total + self.total_delta) if t.total is not None else None,
            line_items=t.line_items, image_path=t.image_path,
        )


def _by_field(report: ExtractionReport):
    return {f.field: f for f in report.fields}


# --- helper-level unit tests ---------------------------------------------------

def test_money_ok_within_and_outside_tolerance():
    assert _money_ok(10.91, 10.90)        # within abs_tol 0.02
    assert _money_ok(100.5, 100.0)        # within rel 0.01 (=1.0)
    assert not _money_ok(11.10, 10.90)    # 0.20 off, outside max(0.02, 1%*10.90=0.109)
    assert not _money_ok(None, 10.90)     # missing prediction is a miss


def test_vendor_ok_fuzzy_and_blank():
    assert _vendor_ok("Trader Joe's", "Trader Joes")  # punctuation/case normalised away
    assert not _vendor_ok("", "Trader Joes")          # blank prediction
    assert not _vendor_ok(None, "Trader Joes")
    assert not _vendor_ok("Walmart", "Trader Joes")


def test_vendor_ok_credits_containment_either_direction():
    # The fuller real name vs WildReceipt's terse Store_name token (ratio is only ~0.57).
    assert _vendor_ok("Costco Wholesale", "COSTCO")
    assert _vendor_ok("COSTCO", "Costco Wholesale")
    # A 1-3 char fragment must NOT match everything (length floor guards against it).
    assert not _vendor_ok("Co", "Costco Wholesale")


# --- harness-level tests -------------------------------------------------------

def test_perfect_extractor_scores_one():
    truths = [_truth("a"), _truth("b", vendor="Costco", total=42.0)]
    report = evaluate_extractor(_PerfectExtractor(truths), truths)
    assert report.n_errors == 0
    assert report.overall == 1.0
    for fa in report.fields:
        assert fa.n == 2 and fa.accuracy == 1.0


def test_broken_extractor_scores_zero_and_counts_errors():
    truths = [_truth("a"), _truth("b")]
    report = evaluate_extractor(_BrokenExtractor(), truths)
    assert report.n_errors == 2
    assert report.overall == 0.0
    for fa in report.fields:
        assert fa.n == 2 and fa.correct == 0


def test_total_within_tolerance_counts_correct():
    truths = [_truth("a")]
    report = evaluate_extractor(_PerturbedExtractor(truths, total_delta=0.01), truths)
    assert _by_field(report)["total"].accuracy == 1.0


def test_total_outside_tolerance_counts_wrong_others_unaffected():
    truths = [_truth("a")]
    report = evaluate_extractor(_PerturbedExtractor(truths, total_delta=5.0), truths)
    fields = _by_field(report)
    assert fields["total"].accuracy == 0.0
    assert fields["vendor"].accuracy == 1.0  # only total was perturbed
    assert fields["subtotal"].accuracy == 1.0


def test_wrong_vendor_only_hurts_vendor():
    truths = [_truth("a")]
    report = evaluate_extractor(_PerturbedExtractor(truths, wrong_vendor=True), truths)
    fields = _by_field(report)
    assert fields["vendor"].accuracy == 0.0
    assert fields["total"].accuracy == 1.0


def test_field_scored_only_when_oracle_has_it():
    # No subtotal, no vendor name, no line items in the oracle -> those aren't scored.
    truths = [_truth("a", vendor="(unknown)", subtotal=None, n_items=0)]
    report = evaluate_extractor(_PerfectExtractor(truths), truths)
    fields = _by_field(report)
    assert fields["subtotal"].n == 0
    assert fields["vendor"].n == 0
    assert fields["line_count"].n == 0
    assert fields["total"].n == 1  # total was present, so it is scored


def test_line_count_requires_exact_match():
    truths = [_truth("a", n_items=3)]

    class _DropsItems(Extractor):
        name = "drops"
        handles = (DocumentType.IMAGE,)

        def extract(self, path, doc_id=None):
            t = truths[0]
            return Receipt(
                doc_id=t.doc_id, vendor_name=t.vendor_name, date=t.date,
                subtotal=t.subtotal, tax_amount=t.tax_amount, total=t.total,
                line_items=t.line_items[:1], image_path=t.image_path,  # under-captured
            )

    report = evaluate_extractor(_DropsItems(), truths)
    assert _by_field(report)["line_count"].accuracy == 0.0


def test_evaluate_extractors_one_report_per_extractor_in_order():
    truths = [_truth("a")]
    reports = evaluate_extractors([_PerfectExtractor(truths), _BrokenExtractor()], truths)
    assert [r.name for r in reports] == ["perfect", "broken"]


def test_overall_is_macro_average_of_scored_fields():
    truths = [_truth("a")]
    report = evaluate_extractor(_PerturbedExtractor(truths, total_delta=5.0), truths)
    scored = [f.accuracy for f in report.fields if f.n]
    assert report.overall == pytest.approx(sum(scored) / len(scored))


def test_report_str_has_overall_and_name():
    truths = [_truth("a")]
    text = str(evaluate_extractor(_PerfectExtractor(truths), truths))
    assert "OVERALL" in text
    assert "perfect" in text
