"""Loader for the WildReceipt corpus (real, legitimate receipts; Apache-2.0).

WildReceipt ships human KIE annotations — each text box has a transcription and a
semantic label (Store_name / Date / Prod_price / Subtotal / Tax / Total / …). Those
annotations act as an **oracle extractor**: we can reconstruct a ``Receipt`` without
OCR and exercise the field detectors on genuine receipts. Because every receipt
here is legitimate, *any* flag is a false positive — so this measures the one thing
the synthetic benchmark cannot: real-world FP rate / detector tolerance.

Get the data (not committed):
    curl -L -o datasets/wildreceipt.tar https://download.openmmlab.com/mmocr/data/wildreceipt.tar
    tar -xf datasets/wildreceipt.tar -C datasets
"""

from __future__ import annotations

import json
import re
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from ..models import DocumentType, LineItem, Receipt

# label id -> field (WildReceipt class_list.txt). Only the ones we consume.
_STORE_NAME, _DATE, _PROD_ITEM, _PROD_PRICE = 1, 7, 11, 15
_SUBTOTAL, _TAX, _TOTAL = 17, 19, 23

_MONEY_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DATE_FORMATS = (
    "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y",
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
    "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%d %b %Y", "%d %B %Y",
)


def _money(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _MONEY_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _parse_date(text: Optional[str]) -> Optional[Date]:
    if not text:
        return None
    s = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _record_to_receipt(rec: dict, doc_id: str) -> Receipt:
    by_label: dict[int, list[str]] = {}
    for ann in rec.get("annotations", []):
        by_label.setdefault(int(ann.get("label", 0)), []).append(str(ann.get("text", "")))

    def first(label: int) -> Optional[str]:
        vals = by_label.get(label)
        return vals[0] if vals else None

    names = by_label.get(_STORE_NAME, [])
    items = by_label.get(_PROD_ITEM, [])
    prices = [p for p in (_money(t) for t in by_label.get(_PROD_PRICE, [])) if p is not None]

    line_items = [
        LineItem(items[i] if i < len(items) else "item", 1, price, price)
        for i, price in enumerate(prices)
    ]

    file_name = rec.get("file_name")
    return Receipt(
        doc_id=doc_id,
        vendor_name=" ".join(names) if names else "(unknown)",
        date=_parse_date(first(_DATE)),
        currency="USD",
        country="US",  # WildReceipt is US receipts -> tax_id detector abstains, as it should
        line_items=line_items,
        subtotal=_money(first(_SUBTOTAL)),
        tax_amount=_money(first(_TAX)),
        total=_money(first(_TOTAL)),
        source=DocumentType.IMAGE,
        image_path=file_name,
    )


def load_receipts(root: Union[str, Path], split: str = "test") -> list[Receipt]:
    """Load WildReceipt receipts from ``root`` (the extracted ``wildreceipt/`` dir).
    ``split`` is ``"train"``, ``"test"`` or ``"both"``."""
    root = Path(root)
    files = {"train": ["train.txt"], "test": ["test.txt"], "both": ["train.txt", "test.txt"]}[split]
    receipts: list[Receipt] = []
    for fname in files:
        path = root / fname
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    receipts.append(_record_to_receipt(json.loads(line), f"{fname}:{i}"))
    return receipts
