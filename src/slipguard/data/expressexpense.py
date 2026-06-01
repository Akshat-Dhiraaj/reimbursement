"""Loader for the ExpressExpense receipt corpus (real receipt *images*; MIT licence).

ExpressExpense's "Large Receipt Image Dataset (SRD)" is 200 real restaurant-receipt
photos under the MIT licence (cite ExpressExpense.com). Unlike WildReceipt and CORD it
ships **no field-level annotations** — just the images. So it cannot be an *oracle*
(there is no ground truth to reconstruct a trusted ``Receipt`` from, and nothing for the
extraction benchmark to score against). Its one honest use is to **broaden the FP audit**:
re-extract each image with a real extractor and audit the detectors on a second, visually
different real-image corpus —

    slipguard eval-real --corpus expressexpense --extractor vlm --limit N

The oracle path (``--extractor oracle``) is meaningless here: with no labelled fields
every detector abstains and the audit is trivially 0 — by design, not a real measurement.

Get the data (not committed; MIT, cite ExpressExpense.com):
    curl -L -o datasets/expressexpense.zip https://expressexpense.com/large-receipt-image-dataset-SRD.zip
    # then unzip into datasets/expressexpense/ (≈19 MB, 200 images)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..models import DocumentType, Receipt

# image suffixes the dataset ships (case-insensitive); kept narrow so we don't pick up
# stray .txt/.md files that sometimes accompany a download.
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})

_FETCH_HINT = (
    "ExpressExpense images not found under {root!r} (not committed). Fetch them (MIT):\n"
    "  curl -L -o datasets/expressexpense.zip "
    "https://expressexpense.com/large-receipt-image-dataset-SRD.zip\n"
    "  # unzip into datasets/expressexpense/"
)


def load_receipts(
    root: Union[str, Path] = Path("datasets") / "expressexpense",
    limit: Optional[int] = None,
) -> list[Receipt]:
    """Return one image-only ``Receipt`` per image found under ``root`` (recursively),
    sorted by path for determinism. Only ``image_path`` is set — there are no labels — so
    these are useful **only** with ``eval-real --extractor vlm/doctr`` (re-extraction),
    never the oracle path. Raises ``FileNotFoundError`` with fetch guidance if no image is
    found, mirroring the WildReceipt loader."""
    root = Path(root)
    images = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(_FETCH_HINT.format(root=str(root)))
    if limit is not None:
        images = images[:limit]

    return [
        Receipt(
            doc_id=f"expressexpense:{p.stem}",
            vendor_name="(unknown)",   # no labels -> oracle detectors abstain; re-extract to score
            date=None,
            source=DocumentType.IMAGE,
            image_path=str(p),
        )
        for p in images
    ]
