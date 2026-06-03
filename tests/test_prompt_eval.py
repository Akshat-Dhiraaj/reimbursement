"""Offline tests for the prompt-accuracy harness (eval/prompt_eval.py).

``validate`` is monkeypatched to return canned verdicts, so the scoring/tally logic is verified with
no network and no API key.
"""

from datetime import date as Date

import pytest

from slipguard.eval import prompt_eval
from slipguard.models import DocumentType, Receipt


def _truth(doc_id, image, vendor, d, sub, tax, total):
    return Receipt(doc_id=doc_id, vendor_name=vendor, date=d, currency="USD", country="US",
                   subtotal=sub, tax_amount=tax, total=total,
                   source=DocumentType.IMAGE, image_path=image)


def test_evaluate_prompt_scores_fields_decisions_and_arith(monkeypatch):
    truths = [
        _truth("t1", "a.png", "Acme Store", Date(2024, 1, 2), 10.0, 0.8, 10.8),
        _truth("t2", "b.png", "Beta Cafe", Date(2024, 2, 3), None, None, 20.0),
        _truth("t3", "c.png", "(unknown)", None, 5.0, 0.5, 5.5),   # unknown vendor + no date → unscored
    ]
    canned = {
        "a.png": {"vendor": "Acme Store", "date": "2024-01-02", "subtotal": 10.0, "tax": 0.8,
                  "total": 10.8, "arithmetic_consistent": True, "decision": "approve"},
        "b.png": {"vendor": "Beta Cafe", "date": "2024-02-03", "subtotal": None, "tax": None,
                  "total": 20.0, "arithmetic_consistent": True, "decision": "approve"},
        "c.png": {"vendor": "Whatever", "date": None, "subtotal": 5.0, "tax": 0.5,
                  "total": 9.9, "arithmetic_consistent": True, "decision": "review"},  # total wrong + bad claim
    }
    monkeypatch.setattr(prompt_eval, "validate", lambda path, **kw: canned[path])

    rep = prompt_eval.evaluate_prompt("dummy.md", truths)
    fa = {f.field: f for f in rep.fields}

    assert rep.name == "dummy"
    assert (fa["vendor"].n, fa["vendor"].correct) == (2, 2)       # t3 vendor unknown → not scored
    assert (fa["date"].n, fa["date"].correct) == (2, 2)           # t3 date None → not scored
    assert (fa["subtotal"].n, fa["subtotal"].correct) == (2, 2)   # t2 subtotal None → not scored
    assert (fa["tax_amount"].n, fa["tax_amount"].correct) == (2, 2)
    assert (fa["total"].n, fa["total"].correct) == (3, 2)         # c.png total 9.9 ≠ 5.5 → miss
    assert rep.overall == pytest.approx((1 + 1 + 1 + 1 + 2 / 3) / 5)

    assert rep.decisions == {"approve": 2, "review": 1, "reject": 0}
    assert rep.approve_rate == pytest.approx(2 / 3)
    # a: subtotal+total present, claim True, truth 10+0.8=10.8 True → agree.
    # b: no subtotal → not checkable. c: claim True, truth 5+0.5=5.5≠9.9 → disagree.
    assert (rep.arith_checked, rep.arith_agree) == (2, 1)
    assert rep.arith_agreement == pytest.approx(0.5)
    assert rep.n_errors == 0 and rep.parse_fails == 0


def test_parse_fail_and_error_are_counted(monkeypatch):
    truths = [
        _truth("t1", "a.png", "Acme", Date(2024, 1, 1), None, None, 5.0),
        _truth("t2", "b.png", "Beta", Date(2024, 1, 2), None, None, 6.0),
    ]

    def fake(path, **kw):
        if path == "a.png":
            return {"decision": "review", "_raw": "not json", "total": 5.0}  # unparseable → _raw stamped
        raise RuntimeError("network down")

    monkeypatch.setattr(prompt_eval, "validate", fake)

    rep = prompt_eval.evaluate_prompt(None, truths)          # None → canonical-prompt label
    assert rep.name == "validity_prompt (default)"
    assert rep.parse_fails == 1
    assert rep.n_errors == 1
    assert rep.decisions["review"] == 1                       # only a.png produced a verdict
    fa = {f.field: f for f in rep.fields}
    assert (fa["total"].n, fa["total"].correct) == (2, 1)     # b.png errored → its total is a miss


def test_evaluate_prompt_aborts_after_consecutive_errors(monkeypatch):
    # A depleted provider quota raises on every call; the harness must bail after
    # max_consecutive_errors instead of grinding through the whole sample (the wasted-60-calls bug).
    truths = [_truth(f"t{i}", f"{i}.png", "Acme", Date(2024, 1, 1), None, None, 5.0) for i in range(8)]
    monkeypatch.setattr(prompt_eval, "validate",
                        lambda path, **kw: (_ for _ in ()).throw(RuntimeError("quota exhausted")))
    rep = prompt_eval.evaluate_prompt("dummy.md", truths, max_consecutive_errors=4)
    assert rep.aborted is True
    assert rep.n_errors == 4          # stopped after 4 in a row, didn't attempt all 8
