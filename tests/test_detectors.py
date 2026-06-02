from datetime import date, timedelta

from slipguard.detectors.arithmetic import ArithmeticConsistencyDetector
from slipguard.detectors.datesanity import DateSanityDetector
from slipguard.detectors.duplicate import DuplicateDetector
from slipguard.detectors.taxid import TaxIdValidationDetector
from slipguard.models import LineItem, Receipt


def clean() -> Receipt:
    items = [LineItem("A", 2, 50.0, 100.0), LineItem("B", 1, 30.0, 30.0)]
    sub = 130.0
    tax = round(0.18 * sub, 2)
    return Receipt(
        "d1", "Croma", date(2026, 1, 10), vendor_tax_id="27AAPFU0939F1ZV",
        line_items=items, subtotal=sub, tax_rate=0.18, tax_amount=tax,
        total=round(sub + tax, 2),
    )


def test_arithmetic_clean_is_low():
    s = ArithmeticConsistencyDetector().score(clean())
    assert not s.abstained and s.score < 0.1


def test_arithmetic_broken_total_is_high():
    r = clean()
    r.total = r.total + 50
    assert ArithmeticConsistencyDetector().score(r).score > 0.6


def test_arithmetic_broken_line_is_high():
    r = clean()
    r.line_items[0].amount = 200.0
    assert ArithmeticConsistencyDetector().score(r).score > 0.6


def test_arithmetic_abstains_without_fields():
    r = Receipt("d", "V", date(2026, 1, 1))
    assert ArithmeticConsistencyDetector().score(r).abstained


def test_arithmetic_service_charge_reconciles():
    # total = subtotal + tax + service charge: modelling the service charge keeps it clean,
    # and without the field the SAME total would read as fraud — proves the field is the fix.
    r = clean()
    r.service_charge = 20.0
    r.total = round(r.subtotal + r.tax_amount + 20.0, 2)
    assert ArithmeticConsistencyDetector().score(r).score < 0.1
    r.service_charge = None
    assert ArithmeticConsistencyDetector().score(r).score > 0.6


def test_arithmetic_discount_reconciles():
    r = clean()
    r.discount = 15.0
    r.total = round(r.subtotal + r.tax_amount - 15.0, 2)
    assert ArithmeticConsistencyDetector().score(r).score < 0.1


def test_arithmetic_tax_inclusive_lines_reconcile():
    # tax-inclusive menu: line prices already include tax, so sum(lines) == subtotal + tax
    # (not the pre-tax subtotal). The detector accepts that instead of flagging subtotal!=lines.
    items = [LineItem("A", 1, 118.0, 118.0)]   # one line, tax-inclusive (== subtotal + tax)
    r = Receipt("ti", "V", date(2026, 1, 10), line_items=items,
                subtotal=100.0, tax_amount=18.0, total=118.0)
    assert ArithmeticConsistencyDetector().score(r).score < 0.1


def test_arithmetic_still_flags_real_mismatch_despite_service():
    # the richer model fixes FPs, it does NOT blanket-suppress a genuine contradiction:
    # a total still wrong after the service charge is accounted for is still caught.
    r = clean()
    r.service_charge = 20.0
    r.total = round(r.subtotal + r.tax_amount + 20.0 + 75.0, 2)  # 75 too high
    assert ArithmeticConsistencyDetector().score(r).score > 0.6


def test_taxid_valid_is_low():
    s = TaxIdValidationDetector().score(clean())
    assert not s.abstained and s.score < 0.1


def test_taxid_invalid_is_high():
    r = clean()
    r.vendor_tax_id = "27AAPFU0939F1Z9"  # known-invalid GSTIN check digit
    assert TaxIdValidationDetector().score(r).score > 0.6


def test_taxid_missing_abstains():
    r = clean()
    r.vendor_tax_id = None
    assert TaxIdValidationDetector().score(r).abstained


def test_taxid_unsupported_country_abstains():
    r = clean()
    r.country = "US"
    assert TaxIdValidationDetector().score(r).abstained


def test_date_future_is_high():
    today = date(2026, 1, 1)
    r = clean()
    r.date = today + timedelta(days=5)
    assert DateSanityDetector(today=today).score(r).score > 0.6


def test_date_past_is_low():
    today = date(2026, 1, 1)
    r = clean()
    r.date = today - timedelta(days=5)
    assert DateSanityDetector(today=today).score(r).score < 0.1


def test_duplicate_detects_resubmission():
    det = DuplicateDetector()
    det.prime([clean()])
    dup = clean()
    dup.doc_id = "d2"  # identical vendor/date/total, new id
    assert det.score(dup).score > 0.6


def test_duplicate_novel_is_low():
    det = DuplicateDetector()
    det.prime([clean()])
    r = clean()
    r.doc_id, r.total, r.date = "d3", 999.99, date(2025, 5, 5)
    assert det.score(r).score < 0.2
