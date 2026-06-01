"""Loader for the CORD corpus (real receipts with structured ground truth; CC-BY-4.0).

CORD (Consolidated Receipt Dataset, naver-clova-ix) ships a human-annotated
``gt_parse`` per receipt — menu lines (name / count / price), a ``sub_total`` block
(subtotal / tax / discount / service) and a ``total`` block. Those labels act as an
**oracle extractor**, exactly like WildReceipt's KIE: we reconstruct a ``Receipt``
without OCR and exercise the field detectors on genuine receipts. Every receipt here
is legitimate, so any flag is a false positive — a *second*, independent real corpus
for the FP audit (WildReceipt is US English; CORD is Indonesian-locale receipts), which
is the one thing the synthetic benchmark cannot measure.

Honest scope, by construction of the labels:
  * CORD's ``gt_parse`` carries **no store name and no date** — so the oracle leaves
    ``vendor_name`` unknown and ``date`` None, and the vendor-less / date_sanity checks
    simply abstain. CORD exercises the **money/arithmetic** reconciliation, not vendor/date.
  * The money fields use Indonesian grouping (``24,000`` / ``60.000`` == 24000 / 60000).
    The shared ``money.parse_money`` reads these correctly because 3 trailing digits mark a
    thousands group, not a decimal — so no per-locale hack is needed (verified in tests).
  * CORD totals legitimately include **service charge / discount** our 3-field model has no
    slot for, so ``total != subtotal + tax`` can fire on a genuine receipt. That is a real
    limitation of the data model (not detector logic) and the FP audit reports it honestly.

Currency is IDR / country ID, so the GSTIN tax-id detector abstains, as it should.

Get the data (not committed; CC-BY-4.0, cite naver-clova-ix/cord-v2):
    # Cached under datasets/cord on first run via the (already-installed) `datasets` lib;
    # `slipguard eval-real --corpus cord` / `eval-extract --corpus cord` fetch it for you.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

from ..models import DocumentType, LineItem, Receipt
from ..money import parse_money as _money  # shared US/EU/grouping-aware parser (see money.py)


def _qty(cnt: object) -> float:
    """First number in a CORD count cell ("2", "x1", "1X" -> 2/1/1); default 1."""
    m = re.search(r"\d+(?:[.,]\d+)?", str(cnt or ""))
    return float(m.group().replace(",", ".")) if m else 1.0


def _line_amount(row: dict) -> Optional[float]:
    """The printed line total for a CORD menu row, mapped faithfully:
    use ``price`` when present, else ``unitprice * cnt`` (some rows give only a unit price),
    then *net any labelled* ``discountprice`` (a negative value on a discounted line — CORD
    lists a discounted item as a duplicate line carrying ``discountprice``; without netting
    it the line is double-counted and ``sum(lines) != subtotal`` fires on a genuine receipt).
    Returns None when there's no money to read."""
    price = _money(row.get("price"))
    if price is None:
        unit = _money(row.get("unitprice"))
        if unit is None:
            return None
        price = round(unit * _qty(row.get("cnt")), 2)
    return round(price + (_money(row.get("discountprice")) or 0.0), 2)

# CORD is the HF dataset id; v2 is the donut-preprocessed release whose `ground_truth`
# is a JSON string holding the structured `gt_parse` we map below.
_HF_DATASET = "naver-clova-ix/cord-v2"


def _menu_lines(menu: object) -> list[LineItem]:
    """Flatten CORD's ``menu`` into line items. ``menu`` is a single dict or a list of
    dicts; each row's ``price`` is the printed *line total*, and a nested ``sub`` holds
    add-on/modifier rows that count toward the subtotal too (so we flatten one level —
    without it ``sum(lines) != subtotal`` on receipts with modifiers). We keep qty=1 and
    unit_price=amount=price, mirroring the WildReceipt oracle: we only trust the line
    total, so the per-line ``qty*unit`` check stays honestly vacuous and the real
    arithmetic signal comes from subtotal/total reconciliation."""
    rows = menu if isinstance(menu, list) else [menu] if isinstance(menu, dict) else []
    items: list[LineItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        amount = _line_amount(row)
        if amount is not None:
            name = str(row.get("nm", "item")).strip() or "item"
            items.append(LineItem(name, 1, amount, amount))
        # add-on rows (a dict or a list of dicts) are separate charges in the subtotal
        sub = row.get("sub")
        for s in (sub if isinstance(sub, list) else [sub] if isinstance(sub, dict) else []):
            sp = _line_amount(s) if isinstance(s, dict) else None
            if sp is not None:
                items.append(LineItem(str(s.get("nm", "item")).strip() or "item", 1, sp, sp))
    return items


def _receipt_from_gt(gt_parse: dict, doc_id: str, image_path: Optional[str] = None) -> Receipt:
    """Map one CORD ``gt_parse`` dict to an oracle ``Receipt``. Pure (no I/O) so the
    mapping is unit-tested directly on hand-built dicts, no network/dataset needed."""
    sub_total = gt_parse.get("sub_total") or {}
    total = gt_parse.get("total") or {}
    if not isinstance(sub_total, dict):
        sub_total = {}
    if not isinstance(total, dict):
        total = {}

    return Receipt(
        doc_id=doc_id,
        vendor_name="(unknown)",          # CORD gt_parse has no store-name label
        date=None,                        # ...nor a date -> date_sanity abstains
        currency="IDR",
        country="ID",                     # so the GSTIN/VAT detector abstains, as it should
        line_items=_menu_lines(gt_parse.get("menu")),
        subtotal=_money(sub_total.get("subtotal_price")),
        tax_amount=_money(sub_total.get("tax_price")),
        total=_money(total.get("total_price")),
        source=DocumentType.IMAGE,
        image_path=image_path,
    )


def _iter_rows(split: str, root: Path, limit: Optional[int]) -> Iterable[dict]:
    """Yield raw CORD rows from the HF dataset, caching under ``root`` (git-ignored).
    Imported lazily so importing this module never requires the `datasets` lib."""
    from datasets import load_dataset  # optional dep; installed with the [vlm] stack

    root.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(_HF_DATASET, split=split, cache_dir=str(root))
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield i, row


def load_receipts(
    root: Union[str, Path] = Path("datasets") / "cord",
    split: str = "test",
    limit: Optional[int] = None,
    save_images: bool = False,
) -> list[Receipt]:
    """Load CORD oracle receipts. ``split`` is ``train``/``validation``/``test``; ``limit``
    caps the count. With ``save_images`` the row image is written under ``root/images`` and
    ``image_path`` set, so a real extractor can re-read it for ``eval-extract`` (the FP
    audit's oracle path needs no image). Raises with fetch guidance if `datasets` is
    missing or the corpus can't be fetched (offline)."""
    import json

    root = Path(root)
    try:
        rows = list(_iter_rows(split, root, limit))
    except ModuleNotFoundError as e:  # the `datasets` lib itself is absent
        raise RuntimeError(
            "CORD needs the `datasets` library (ships with the [vlm] extra): "
            f'{e}. Install it: pip install -e ".[vlm]"'
        ) from e

    img_dir = root / "images" / split
    if save_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    receipts: list[Receipt] = []
    for i, row in rows:
        gt_parse = json.loads(row["ground_truth"]).get("gt_parse", {}) or {}
        image_path = None
        if save_images:
            # CORD rows carry a PIL image; persist it once so the extractor has a file.
            path = img_dir / f"cord-{split}-{i:04d}.png"
            if not path.exists():
                row["image"].save(path)
            image_path = str(path)
        receipts.append(_receipt_from_gt(gt_parse, f"cord-{split}:{i}", image_path))
    return receipts
