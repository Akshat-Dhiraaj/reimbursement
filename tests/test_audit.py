from datetime import date

from slipguard.detectors import default_detectors
from slipguard.detectors.duplicate import DuplicateDetector
from slipguard.eval.audit import audit_false_positives
from slipguard.fusion import Fuser
from slipguard.models import DocumentType, LineItem, Receipt


def _clean(doc_id: str, total: float = 130.0) -> Receipt:
    items = [LineItem("A", 2, 50.0, 100.0), LineItem("B", 1, 30.0, 30.0)]
    return Receipt(doc_id, "Croma", date(2026, 1, 10),
                   line_items=items, subtotal=130.0, total=total)


def test_audit_clean_corpus_has_zero_fp():
    receipts = [_clean(f"d{i}") for i in range(5)]
    audit = audit_false_positives(receipts, default_detectors(), Fuser())
    assert audit.n == 5
    assert audit.fp_rate == 0.0
    assert audit.decisions["approve"] == 5


def test_audit_flags_broken_arithmetic():
    audit = audit_false_positives([_clean("bad", total=999.0)], default_detectors(), Fuser())
    assert audit.fp_rate == 1.0
    arith = next(d for d in audit.detectors if d.name == "arithmetic")
    assert arith.n_flag == 1


def test_audit_records_field_coverage():
    audit = audit_false_positives([_clean("d0")], default_detectors(), Fuser())
    assert audit.field_coverage["total"] == 1
    assert audit.field_coverage["line_items"] == 1


def test_audit_tolerates_receipt_without_date():
    # real data exposed this: the duplicate detector crashed on a None date
    r = Receipt("nd", "Vendor", None, source=DocumentType.IMAGE,
                line_items=[LineItem("x", 1, 4.0, 4.0)], total=4.0)
    audit = audit_false_positives([r], default_detectors(), Fuser())
    assert audit.n == 1  # did not raise


def test_duplicate_handles_none_date():
    det = DuplicateDetector()
    det.prime([])
    sig = det.score(Receipt("x", "V", None, total=10.0))  # must not raise
    assert sig.score < 0.2
