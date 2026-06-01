"""Synthetic labelled receipts with field-level ground truth.

Generates internally-consistent clean receipts, then mints matched fraud
variants that each violate exactly one rule (arithmetic, date, tax-id, duplicate).
This gives a controlled, reproducible benchmark with known fraud subtypes — the
backbone the harness ranks detectors on. AI-generated / image-tamper subtypes are
declared in the label space but require rendered images (image route, later).

NOTE: strong numbers here only validate the harness + deterministic layer on
*synthetic* fraud that violates these exact rules. Real-world performance against
subtle / AI-generated fraud must be measured on real corpora and the image route.
"""

from __future__ import annotations

import copy
import random
import string
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from typing import Optional

from stdnum.in_ import gstin

from ..models import FraudType, LabeledSample, LineItem, Receipt

_ALNUM = string.digits + string.ascii_uppercase


@dataclass
class _Vendor:
    name: str
    state: str  # GST state code (first 2 digits of GSTIN)
    tax_rate: float
    items: list[tuple[str, float, float]]  # (description, min_price, max_price)


_VENDORS = [
    _Vendor("Reliance Fresh", "27", 0.05,
            [("Atta 5kg", 200, 320), ("Milk 1L", 50, 70), ("Rice 5kg", 300, 520),
             ("Cooking Oil 1L", 120, 220), ("Eggs 12", 60, 95)]),
    _Vendor("Croma", "27", 0.18,
            [("USB-C Cable", 300, 900), ("Power Bank", 1200, 3500),
             ("Bluetooth Speaker", 1500, 6000), ("Mouse", 500, 2500)]),
    _Vendor("Apollo Pharmacy", "29", 0.12,
            [("Paracetamol", 20, 60), ("Vitamin D3", 150, 400),
             ("Bandage Pack", 40, 120), ("Cough Syrup", 90, 240)]),
    _Vendor("Cafe Coffee Day", "07", 0.05,
            [("Cappuccino", 120, 220), ("Sandwich", 150, 320), ("Cold Coffee", 160, 280)]),
    _Vendor("Big Bazaar", "09", 0.18,
            [("Notebook", 40, 120), ("Detergent 1kg", 110, 260),
             ("Shampoo", 90, 320), ("Toothpaste", 45, 130)]),
]


@dataclass
class Dataset:
    history: list[Receipt]  # legitimate prior submissions (dup reference / future train)
    samples: list[LabeledSample] = field(default_factory=list)


def _make_valid_gstin(rng: random.Random, state: str) -> str:
    for _ in range(200):
        pan = ("".join(rng.choice(string.ascii_uppercase) for _ in range(5))
               + "".join(rng.choice(string.digits) for _ in range(4))
               + rng.choice(string.ascii_uppercase))
        base = state + pan + "1" + "Z"  # 14 chars; brute-force the 15th check char
        for c in _ALNUM:
            if gstin.is_valid(base + c):
                return base + c
    raise RuntimeError("could not construct a valid GSTIN")


def _corrupt_gstin(rng: random.Random, g: str) -> str:
    for _ in range(50):
        chars = list(g)
        i = rng.randrange(len(chars))
        new = rng.choice(_ALNUM)
        if new == chars[i]:
            continue
        chars[i] = new
        cand = "".join(chars)
        if not gstin.is_valid(cand):
            return cand
    return g + "X"  # always invalid (length 16)


def _clean_receipt(rng: random.Random, doc_id: str, today: Date) -> Receipt:
    vendor = rng.choice(_VENDORS)
    n = rng.randint(1, 5)
    items: list[LineItem] = []
    for _ in range(n):
        desc, lo, hi = rng.choice(vendor.items)
        qty = rng.randint(1, 4)
        price = round(rng.uniform(lo, hi), 2)
        items.append(LineItem(desc, qty, price, round(qty * price, 2)))
    subtotal = round(sum(li.amount for li in items), 2)
    tax = round(vendor.tax_rate * subtotal, 2)
    total = round(subtotal + tax, 2)
    receipt_date = today - timedelta(days=rng.randint(1, 180))
    return Receipt(
        doc_id=doc_id, vendor_name=vendor.name, date=receipt_date,
        currency="INR", country="IN", vendor_tax_id=_make_valid_gstin(rng, vendor.state),
        line_items=items, subtotal=subtotal, tax_rate=vendor.tax_rate,
        tax_amount=tax, total=total,
    )


def _tamper_arithmetic(rng: random.Random, r: Receipt) -> dict:
    mode = rng.choice(["total", "line", "tax"])
    if mode == "total":
        old = r.total
        r.total = round(r.total * rng.uniform(1.1, 1.4), 2)
        return {"mode": "total", "from": old, "to": r.total}
    if mode == "line" and r.line_items:
        i = rng.randrange(len(r.line_items))
        old = r.line_items[i].amount
        r.line_items[i].amount = round(old * rng.uniform(1.2, 2.0), 2)
        return {"mode": "line", "index": i, "from": old, "to": r.line_items[i].amount}
    old = r.tax_amount
    r.tax_amount = round((r.tax_amount or 1.0) * rng.uniform(1.5, 3.0) + 1.0, 2)
    return {"mode": "tax", "from": old, "to": r.tax_amount}


def _tamper_date(rng: random.Random, r: Receipt, today: Date) -> dict:
    old = r.date
    r.date = today + timedelta(days=rng.randint(1, 60))
    return {"from": old.isoformat(), "to": r.date.isoformat()}


def _tamper_tax_id(rng: random.Random, r: Receipt) -> dict:
    old = r.vendor_tax_id
    r.vendor_tax_id = _corrupt_gstin(rng, r.vendor_tax_id or "27AAPFU0939F1ZV")
    return {"from": old, "to": r.vendor_tax_id}


def generate(
    n_clean: int = 120,
    fraud_per_type: int = 30,
    n_history: int = 80,
    seed: int = 0,
    today: Optional[Date] = None,
) -> Dataset:
    """Build a reproducible labelled benchmark.

    Returns a :class:`Dataset` with ``history`` (legitimate prior submissions used
    as the duplicate reference) and ``samples`` (clean + fraud eval items)."""

    rng = random.Random(seed)
    today = today or Date.today()

    history = [_clean_receipt(rng, f"hist-{i:04d}", today) for i in range(n_history)]

    samples: list[LabeledSample] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"ev-{counter:04d}"

    for _ in range(n_clean):
        r = _clean_receipt(rng, next_id(), today)
        samples.append(LabeledSample(r, is_fraud=False, fraud_types={FraudType.NONE}))

    for _ in range(fraud_per_type):
        r = _clean_receipt(rng, next_id(), today)
        detail = _tamper_arithmetic(rng, r)
        samples.append(LabeledSample(r, True, {FraudType.ARITHMETIC}, detail))

    for _ in range(fraud_per_type):
        r = _clean_receipt(rng, next_id(), today)
        detail = _tamper_date(rng, r, today)
        samples.append(LabeledSample(r, True, {FraudType.DATE}, detail))

    for _ in range(fraud_per_type):
        r = _clean_receipt(rng, next_id(), today)
        detail = _tamper_tax_id(rng, r)
        samples.append(LabeledSample(r, True, {FraudType.TAX_ID}, detail))

    for _ in range(fraud_per_type):
        original = rng.choice(history)
        clone = copy.deepcopy(original)
        clone.doc_id = next_id()
        samples.append(LabeledSample(clone, True, {FraudType.DUPLICATE},
                                     {"original": original.doc_id}))

    rng.shuffle(samples)
    return Dataset(history=history, samples=samples)
