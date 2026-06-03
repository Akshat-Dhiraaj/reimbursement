"""Synthetic image-EXIF benchmark for the provenance route — the IMAGE sibling of
:mod:`slipguard.data.pdfsynth`.

Mints tiny but real JPEGs on disk with genuine EXIF, clean ones and matched tampers
that each carry exactly one provenance defect:

* ``editor_tag``     — EXIF ``Software`` naming an image editor (Photoshop, GIMP, …)
* ``date_mismatch``  — ``DateTime`` (last write) well after ``DateTimeOriginal`` (capture)

There is deliberately **no** "incremental update" analogue here: that is a PDF
structural artefact with no clean EXIF counterpart, and we do not invent a weak third
signal just to match the PDF bench. Each fraud isolates a single defect so the
per-signal behaviour is unambiguous.

Pillow (the ``[vlm]`` extra) is required to *write* EXIF, so it is imported lazily
inside :func:`generate_image`; importing this module stays dependency-free.
"""

from __future__ import annotations

import random
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from ..models import DocumentType, FraudType, LabeledSample, Receipt
from ._common import (VENDOR_NAMES, event_datetime, image_or_pdf_receipt,
                      mismatched_modified)
from .synth import Dataset

# (Make, Model) pairs for genuine phone captures.
_CAMERAS = [
    ("Apple", "iPhone 12"), ("Apple", "iPhone 13"), ("samsung", "SM-G991B"),
    ("Google", "Pixel 6"), ("OnePlus", "ONEPLUS A6013"), ("Xiaomi", "M2101K6G"),
]
#: Software strings a phone capture pipeline writes — NONE of these are image editors,
#: so a clean/date-mismatch sample carries no spurious editor tag.
_CAMERA_SOFTWARE = [None, "12.5.1", "HDR+ 1.0.540170750", "G991BXXU3AUH1", "MIUI 12.5.2"]
#: Software strings written by raster/photo editors (each matches KNOWN_IMAGE_EDITORS).
_EDITOR_SOFTWARE = [
    "Adobe Photoshop 24.1 (Windows)", "GIMP 2.10.34", "Snapseed 2.21",
    "Pixelmator Pro 3.3", "PicsArt 23.1.0", "Canva", "Photopea 1.0",
]


def _exif_dt(dt: datetime) -> str:
    return dt.strftime("%Y:%m:%d %H:%M:%S")


def _build_exif(
    software: Optional[str], make: str, model: str,
    captured: datetime, modified: datetime,
):
    """Construct a Pillow ``Exif`` with capture/modify times, camera make/model and an
    optional Software tag. ``DateTimeOriginal`` lives in the Exif sub-IFD, exactly where
    a real camera writes it (verified to round-trip through ``inspect_image``)."""
    from PIL import Image

    exif = Image.Exif()
    exif[0x0132] = _exif_dt(modified)   # DateTime — last write
    exif[0x010F] = make                 # Make
    exif[0x0110] = model                # Model
    if software:
        exif[0x0131] = software          # Software
    sub = exif.get_ifd(0x8769)           # Exif sub-IFD
    sub[0x9003] = _exif_dt(captured)     # DateTimeOriginal — capture
    return exif


def _write_image(workdir: Path, doc_id: str, exif) -> Path:
    from PIL import Image

    path = workdir / f"{doc_id}.jpg"
    Image.new("RGB", (32, 32), "white").save(path, "JPEG", exif=exif)
    return path


def _receipt(rng: random.Random, doc_id: str, path: Path, today: Date) -> Receipt:
    return image_or_pdf_receipt(rng, doc_id, path, today,
                                source=DocumentType.IMAGE, image_path=str(path))


def _captured_at(rng: random.Random, today: Date) -> datetime:
    return event_datetime(rng, today)


def generate_image(
    n_clean: int = 40,
    fraud_per_type: int = 15,
    seed: int = 0,
    today: Optional[Date] = None,
    workdir: Union[str, Path, None] = None,
) -> Dataset:
    """Build a reproducible image-EXIF provenance benchmark, writing real JPEGs under
    ``workdir`` and returning a :class:`Dataset` of labelled IMAGE samples. Requires
    Pillow (the ``[vlm]`` extra)."""
    rng = random.Random(seed)
    today = today or Date.today()
    workdir = Path(workdir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    samples: list[LabeledSample] = []
    counter = 0

    def make(is_fraud: bool, ftypes: set, exif, detail: dict) -> None:
        nonlocal counter
        counter += 1
        doc_id = f"img-{counter:04d}"
        path = _write_image(workdir, doc_id, exif)
        receipt = _receipt(rng, doc_id, path, today)
        samples.append(LabeledSample(receipt, is_fraud, ftypes, detail))

    for _ in range(n_clean):
        captured = _captured_at(rng, today)
        make_, model = rng.choice(_CAMERAS)
        exif = _build_exif(rng.choice(_CAMERA_SOFTWARE), make_, model, captured, captured)
        make(False, {FraudType.NONE}, exif, {})

    for _ in range(fraud_per_type):
        captured = _captured_at(rng, today)
        make_, model = rng.choice(_CAMERAS)
        software = rng.choice(_EDITOR_SOFTWARE)
        # editor tag is the only defect: capture == modify, so no date signal
        exif = _build_exif(software, make_, model, captured, captured)
        make(True, {FraudType.METADATA}, exif, {"mode": "editor_tag", "software": software})

    for _ in range(fraud_per_type):
        captured = _captured_at(rng, today)
        make_, model = rng.choice(_CAMERAS)
        modified, gap = mismatched_modified(rng, captured)
        # date mismatch is the only defect: a camera (non-editor) Software tag
        exif = _build_exif(rng.choice(_CAMERA_SOFTWARE), make_, model, captured, modified)
        make(True, {FraudType.METADATA}, exif,
             {"mode": "date_mismatch", "gap_days": gap})

    rng.shuffle(samples)
    return Dataset(history=[], samples=samples)
