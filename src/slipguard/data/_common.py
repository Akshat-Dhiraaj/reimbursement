"""Shared synthesis helpers for the data generators (``synth`` / ``pdfsynth`` / ``imagesynth``) and
the tamper tools — single-sourced so the three benchmarks can't drift.

Every RNG helper makes its draws in a FIXED order; the seeded eval tests (``test_pdf_forensics``,
``test_image_forensics``, ``test_pdf_deep_forensics``, ``test_synth`` …) pin ``seed=0`` and depend on
that order, so preserve the exact call sequence if you edit these."""

from __future__ import annotations

import random
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..models import DocumentType, Receipt

#: Receipt vendors for the synthetic image/PDF corpora (names only; ``synth.py`` keeps a richer
#: ``_Vendor`` list whose names mirror these).
VENDOR_NAMES = ["Reliance Fresh", "Croma", "Apollo Pharmacy", "Cafe Coffee Day", "Big Bazaar"]

#: Image globs the tamper tools scan for source receipts.
_SRC_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def receipt_date(rng: random.Random, today: Date) -> Date:
    """A receipt's business date — within the 60-day reimbursement window, so ``date_sanity`` stays
    quiet on clean samples. One rng draw."""
    return today - timedelta(days=rng.randint(1, 55))


def event_datetime(rng: random.Random, today: Date) -> datetime:
    """A capture/issue timestamp (EXIF ``DateTimeOriginal`` / PDF ``CreationDate``): a day in the
    last ~6 months at a daytime hour. Three rng draws (day, hour, minute)."""
    day = today - timedelta(days=rng.randint(1, 180))
    return datetime(day.year, day.month, day.day, rng.randint(8, 20), rng.randint(0, 59))


def mismatched_modified(rng: random.Random, base: datetime) -> tuple[datetime, int]:
    """A "modified long after issued/captured" timestamp + the gap in days — the metadata
    date-mismatch fraud signal. One rng draw."""
    modified = base + timedelta(days=rng.randint(15, 400))
    return modified, (modified - base).days


def image_or_pdf_receipt(rng: random.Random, doc_id: str, path, today: Date, *,
                         source: DocumentType, image_path: Optional[str] = None) -> Receipt:
    """A minimal Receipt for the forensics corpora — the business fields live in the rendered file,
    not the model, so only a vendor + an in-window date are set. Two rng draws (vendor, then date)."""
    return Receipt(
        doc_id=doc_id,
        vendor_name=rng.choice(VENDOR_NAMES),
        date=receipt_date(rng, today),
        source=source,
        source_path=str(path),
        image_path=image_path,
    )


def list_source_images(src_dir, limit: Optional[int] = None) -> list[Path]:
    """Source receipt images under ``src_dir`` (sorted for determinism), capped at ``limit``."""
    srcs = sorted(p for g in _SRC_GLOBS for p in Path(src_dir).glob(g))
    return srcs[:limit] if limit else srcs
