from datetime import date

from stdnum.in_ import gstin

from slipguard.data.synth import generate
from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.models import FraudType

TODAY = date(2026, 6, 1)


def test_generate_is_reproducible():
    a = generate(seed=1)
    b = generate(seed=1)
    assert [s.receipt.doc_id for s in a.samples] == [s.receipt.doc_id for s in b.samples]
    assert [s.receipt.total for s in a.samples] == [s.receipt.total for s in b.samples]


def test_clean_samples_reconcile_and_validate():
    ds = generate(seed=2, today=TODAY)
    arith = ArithmeticConsistencyDetector()
    clean = [s for s in ds.samples if not s.is_fraud]
    assert clean
    for s in clean:
        assert arith.score(s.receipt).score < 0.1
        assert gstin.is_valid(s.receipt.vendor_tax_id)
        assert s.receipt.date <= TODAY


def test_fraud_subtypes_violate_their_rule():
    ds = generate(seed=3, today=TODAY)
    arith = ArithmeticConsistencyDetector()
    seen = set()
    for s in ds.samples:
        seen |= s.fraud_types
        if FraudType.TAX_ID in s.fraud_types:
            assert not gstin.is_valid(s.receipt.vendor_tax_id)
        if FraudType.DATE in s.fraud_types:
            assert s.receipt.date > TODAY
        if FraudType.ARITHMETIC in s.fraud_types:
            assert arith.score(s.receipt).score > 0.5
    assert {FraudType.ARITHMETIC, FraudType.DATE, FraudType.TAX_ID, FraudType.DUPLICATE} <= seen


def test_duplicates_reference_history():
    ds = generate(seed=4)
    hist_ids = {r.doc_id for r in ds.history}
    dups = [s for s in ds.samples if FraudType.DUPLICATE in s.fraud_types]
    assert dups
    for s in dups:
        assert s.detail["original"] in hist_ids
