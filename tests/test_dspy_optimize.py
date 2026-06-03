"""Offline tests for the DSPy dev-time optimizer (dspy_optimize.py).

The optimize/compare run needs network + GROQ_API_KEY, so it isn't exercised here. We pin the
pure pieces — the field-accuracy metric (shared with eval/extraction), the prediction→Receipt
mapping, and the BootstrapFewShot metric's pass/fail — plus that the DSPy signature builds. The
live comparison is run by hand: `python -m slipguard.dspy_optimize`.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from slipguard import dspy_optimize as D
from slipguard.models import LineItem, Receipt


def _r(vendor="Croma", d=date(2026, 1, 10), sub=100.0, tax=18.0, tot=118.0, n=2) -> Receipt:
    return Receipt("x", vendor, d, line_items=[LineItem("i", 1, 0, 0)] * n,
                   subtotal=sub, tax_amount=tax, total=tot)


def test_field_score_perfect_and_poor():
    gold = _r()
    assert D.receipt_field_score(gold, _r()) == 1.0
    poor = _r(vendor="Zzz Unrelated", d=date(2025, 1, 1), sub=1, tax=1, tot=1, n=9)
    assert D.receipt_field_score(gold, poor) < 0.3


def test_pred_to_receipt_maps_typed_fields():
    pred = SimpleNamespace(vendor="Costco", date="2026-01-10", subtotal=100.0,
                           tax=8.0, total=108.0, line_count=3)
    r = D.pred_to_receipt(pred, "d", "p.jpg")
    assert r.vendor_name == "Costco" and r.date == date(2026, 1, 10)
    assert r.total == 108.0 and r.tax_amount == 8.0 and len(r.line_items) == 3


def test_num_treats_zero_as_absent():
    assert D._num(0) is None and D._num("0") is None      # signature uses 0 for "not shown"
    assert D._num(58.22) == 58.22 and D._num("x") is None


def test_bootstrap_metric_pass_fail():
    # a faithful prediction passes the demo threshold; a wrong one fails
    ex = SimpleNamespace(vendor="Croma", date="2026-01-10", subtotal=100.0, tax=18.0,
                         total=118.0, line_count=2)
    good = SimpleNamespace(vendor="Croma", date="2026-01-10", subtotal=100.0, tax=18.0,
                           total=118.0, line_count=2)
    bad = SimpleNamespace(vendor="Nope", date="", subtotal=0, tax=0, total=0, line_count=0)
    assert D._metric(ex, good) is True
    assert D._metric(ex, bad) is False


def test_signature_builds():
    pytest.importorskip("dspy")
    Sig = D.make_signature()
    assert "image" in Sig.input_fields
    for f in ("vendor", "date", "subtotal", "tax", "total", "line_count"):
        assert f in Sig.output_fields
