from datetime import date

from slipguard.detectors import default_detectors
from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.detectors.duplicate import DuplicateDetector
from slipguard.eval.audit import audit_false_positives, image_bearing, reextract
from slipguard.extractors.base import Extractor
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


class _StubExtractor(Extractor):
    """Stands in for a real VLM so the reextract wiring is testable without a GPU.
    Echoes the path/doc_id it was handed and reports a low line-item parse
    confidence (as a faithful but under-capturing extractor would)."""

    name = "stub"
    handles = (DocumentType.IMAGE,)

    def extract(self, path, doc_id=None):
        return Receipt(doc_id or path, "Stub Vendor", date(2026, 1, 10),
                       source=DocumentType.IMAGE, image_path=path,
                       line_items=[LineItem("kept", 1, 2.0, 2.0)],
                       subtotal=100.0, total=100.0,
                       field_confidence={"line_items": 0.2})


def test_image_bearing_selects_same_subset_oracle_and_reextract_compare_on():
    # The oracle-limit audit and a re-extraction run must see the IDENTICAL receipts for
    # the FP comparison to be apples-to-apples; both go through image_bearing.
    receipts = [
        Receipt("r0", "V", date(2026, 1, 10), image_path="/img/0.jpg"),
        Receipt("r1", "V", date(2026, 1, 10), image_path=None),   # no image -> excluded
        Receipt("r2", "V", date(2026, 1, 10), image_path="/img/2.jpg"),
    ]
    assert [r.doc_id for r in image_bearing(receipts)] == ["r0", "r2"]
    assert [r.doc_id for r in image_bearing(receipts, limit=1)] == ["r0"]
    # same selection reextract feeds the extractor -> oracle/re-extract see the same N
    assert [r.doc_id for r in image_bearing(receipts, 1)] == \
           [r.doc_id for r in reextract(_StubExtractor(), receipts, limit=1)]


def test_reextract_skips_imageless_and_respects_limit():
    receipts = [
        Receipt("r0", "V", date(2026, 1, 10), image_path="/img/0.jpg"),
        Receipt("r1", "V", date(2026, 1, 10), image_path=None),  # no image -> skipped
        Receipt("r2", "V", date(2026, 1, 10), image_path="/img/2.jpg"),
    ]
    out = reextract(_StubExtractor(), receipts)
    assert [r.doc_id for r in out] == ["r0", "r2"]  # image-less r1 dropped

    capped = reextract(_StubExtractor(), receipts, limit=1)
    assert [r.doc_id for r in capped] == ["r0"]


def test_audit_guard_is_what_prevents_the_false_positive():
    # A legitimate receipt whose subtotal disagrees with its (under-captured) lines:
    # arithmetic WOULD cry fraud, but the low line-item parse confidence should make
    # it abstain. Proves the abstain guard -- not luck -- is what averts the FP.
    r = Receipt("fp", "Vendor", date(2026, 1, 10),
                line_items=[LineItem("x", 1, 2.0, 2.0)],
                subtotal=100.0, total=100.0,
                field_confidence={"line_items": 0.2})

    guarded = audit_false_positives([r], [ArithmeticConsistencyDetector(min_confidence=0.5)], Fuser())
    assert guarded.fp_rate == 0.0  # guard armed -> abstains -> no false positive

    unguarded = audit_false_positives([r], [ArithmeticConsistencyDetector(min_confidence=0.0)], Fuser())
    assert unguarded.fp_rate > 0.0  # guard disabled -> the same mismatch is (wrongly) flagged
