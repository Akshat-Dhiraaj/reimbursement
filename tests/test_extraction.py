import json
from datetime import date
from pathlib import Path

from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.extractors import StructuredExtractor, default_extractors, extractor_for
from slipguard.extractors.base import Extractor
from slipguard.models import DocumentType, LineItem, Receipt


def _clean() -> Receipt:
    items = [LineItem("A", 2, 50.0, 100.0), LineItem("B", 1, 30.0, 30.0)]
    sub = 130.0
    tax = round(0.18 * sub, 2)
    return Receipt(
        "d1", "Croma", date(2026, 1, 10),
        line_items=items, subtotal=sub, tax_rate=0.18, tax_amount=tax,
        total=round(sub + tax, 2),
    )


# ---- Extractor interface + registry ---------------------------------------

def test_structured_extractor_roundtrips_json(tmp_path: Path):
    payload = {
        "doc_id": "x1", "vendor_name": "Croma", "date": "2026-01-10",
        "line_items": [{"description": "A", "quantity": 2, "unit_price": 50.0, "amount": 100.0}],
        "subtotal": 100.0, "tax_rate": 0.18, "tax_amount": 18.0, "total": 118.0,
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    r = StructuredExtractor().extract(str(p))
    assert isinstance(r, Receipt)
    assert r.doc_id == "x1" and r.vendor_name == "Croma"
    assert r.date == date(2026, 1, 10)
    assert len(r.line_items) == 1 and r.line_items[0].amount == 100.0
    assert r.total == 118.0


def test_structured_extractor_handles_only_structured():
    ex = StructuredExtractor()
    assert ex.can_handle(DocumentType.STRUCTURED)
    assert not ex.can_handle(DocumentType.PDF)
    assert not ex.can_handle(DocumentType.IMAGE)


def test_extractor_for_selects_by_route():
    assert isinstance(extractor_for(DocumentType.STRUCTURED), StructuredExtractor)
    # PDF / IMAGE have no extractor wired yet -> None (honest "not wired" signal)
    assert extractor_for(DocumentType.PDF) is None
    assert extractor_for(DocumentType.IMAGE) is None


def test_default_extractors_are_extractors():
    assert default_extractors()
    assert all(isinstance(e, Extractor) for e in default_extractors())


# ---- Arithmetic low-confidence abstain guard ------------------------------

def test_arithmetic_unchanged_without_confidence_info():
    # Empty field_confidence == trusted: behaviour identical to before the guard.
    s = ArithmeticConsistencyDetector().score(_clean())
    assert not s.abstained and s.score < 0.1


def test_arithmetic_abstains_on_low_confidence_field():
    r = _clean()
    r.total = r.total + 50              # would normally flag as a mismatch
    r.field_confidence = {"total": 0.2}
    s = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert s.abstained                  # likely a misread, not fraud -> abstain


def test_arithmetic_flags_when_confidence_high():
    r = _clean()
    r.total = r.total + 50
    r.field_confidence = {"total": 0.95}   # confidently read AND inconsistent -> fraud
    s = ArithmeticConsistencyDetector(min_confidence=0.5).score(r)
    assert not s.abstained and s.score > 0.6
