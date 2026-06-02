"""Tests for the learned logistic fuser (fusion_learned.py) and the pluggable
``Fuser.combiner`` it slots into.

These are pure and fast — no real corpora, no synthetic-dataset run. They pin the
contract (feature ordering, abstain handling), the optimiser (separable data,
determinism), the integration with ``Fuser`` (default still noisy-OR), and — most
importantly — the *mechanism* that justifies the whole milestone: a noisy detector
that fires on negatives too gets a smaller learned weight than a clean one.
"""

from __future__ import annotations

from slipguard.combine import noisy_or
from slipguard.fusion import Fuser
from slipguard.fusion_learned import (
    LearnedFuser,
    feature_names,
    feature_vector,
    fit_logistic,
)
from slipguard.models import Signal


def _sig(name: str, score: float, conf: float) -> Signal:
    return Signal(detector=name, score=score, confidence=conf)


# --- feature vector ----------------------------------------------------------

def test_feature_names_are_the_canonical_detectors():
    names = feature_names()
    assert names == [
        "arithmetic", "tax_id", "date_sanity", "duplicate", "pdf_meta", "image_meta",
    ]


def test_feature_vector_orders_by_name_regardless_of_signal_order():
    order = ["a", "b", "c"]
    # signals supplied out of order; one detector ("c") missing entirely
    signals = [_sig("b", 0.5, 1.0), _sig("a", 0.8, 0.5)]
    assert feature_vector(signals, order) == [0.4, 0.5, 0.0]  # a=0.8*0.5, b=0.5*1, c missing->0


def test_feature_vector_zeros_abstainers():
    # confidence 0 == abstain -> weighted 0, so it contributes nothing (same as noisy-OR).
    order = ["a", "b"]
    signals = [_sig("a", 0.9, 0.0), _sig("b", 0.7, 1.0)]
    assert feature_vector(signals, order) == [0.0, 0.7]


# --- the optimiser ----------------------------------------------------------

def test_fit_logistic_separates_linearly_separable_data():
    # one feature; positives have it high, negatives low -> learned prob must order them.
    X = [[1.0]] * 20 + [[0.0]] * 20
    y = [1.0] * 20 + [0.0] * 20
    w, b = fit_logistic(X, y, iters=2000)
    assert w[0] > 0.0
    from slipguard.fusion_learned import _sigmoid
    assert _sigmoid(b + w[0] * 1.0) > 0.5 > _sigmoid(b + w[0] * 0.0)


def test_fit_is_deterministic():
    X = [[0.3, 0.1], [0.8, 0.0], [0.0, 0.9], [0.2, 0.2]]
    y = [0.0, 1.0, 1.0, 0.0]
    assert fit_logistic(X, y, iters=500) == fit_logistic(X, y, iters=500)


def test_learned_downweights_a_noisy_detector():
    """The crux of the milestone. Detector 'clean' fires only on fraud; detector
    'noisy' fires on fraud AND on legitimate receipts (the arithmetic-on-lossy-
    extraction pattern). The learned fuser must trust 'clean' more than 'noisy'."""
    order = ["noisy", "clean"]
    rows = []
    # positives: both fire
    for _ in range(40):
        rows.append(([_sig("noisy", 0.8, 1.0), _sig("clean", 0.8, 1.0)], True))
    # negatives: only the noisy detector fires (clean stays silent)
    for _ in range(40):
        rows.append(([_sig("noisy", 0.8, 1.0), _sig("clean", 0.0, 1.0)], False))

    fuser = LearnedFuser.fit(rows, order)
    w = dict(zip(fuser.detector_order, fuser.weights))
    assert w["clean"] > w["noisy"]  # the discriminating signal earns more weight
    # and the fuser scores a fraud-shaped input above a legit-shaped one
    fraud = [_sig("noisy", 0.8, 1.0), _sig("clean", 0.8, 1.0)]
    legit = [_sig("noisy", 0.8, 1.0), _sig("clean", 0.0, 1.0)]
    assert fuser(fraud) > fuser(legit)


# --- integration with Fuser --------------------------------------------------

def test_fuser_default_is_unchanged_noisy_or():
    # guard the refactor: with no combiner, risk == noisy-OR over non-abstained weighted.
    signals = [_sig("a", 0.6, 1.0), _sig("b", 0.5, 0.4), _sig("c", 0.9, 0.0)]
    expected = noisy_or([0.6, 0.2])  # c abstains (conf 0) -> dropped
    assert Fuser().risk(signals) == expected


def test_fuser_combiner_overrides_and_is_clamped():
    signals = [_sig("a", 0.6, 1.0)]
    assert Fuser(combiner=lambda s: 0.123).risk(signals) == 0.123
    assert Fuser(combiner=lambda s: 5.0).risk(signals) == 1.0   # clamp high
    assert Fuser(combiner=lambda s: -2.0).risk(signals) == 0.0  # clamp low


def test_learned_fuser_plugs_into_fuser_verdict():
    fuser = LearnedFuser(detector_order=["a"], weights=[10.0], bias=-1.0)
    f = Fuser(combiner=fuser)
    # a strong 'a' signal drives risk high enough to reject
    v = f.verdict("doc1", [_sig("a", 1.0, 1.0)])
    assert v.risk_score > 0.85 and v.decision.value == "reject"


def test_explain_sorts_by_magnitude_and_includes_bias():
    fuser = LearnedFuser(detector_order=["a", "b", "c"], weights=[0.1, -2.0, 0.5], bias=0.3)
    keys = list(fuser.explain())
    assert keys == ["b", "c", "a", "_bias"]  # |−2.0| > |0.5| > |0.1|
    assert fuser.explain()["_bias"] == 0.3
