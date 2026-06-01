"""Synthetic PDF benchmark for the provenance route.

Mints tiny but structurally valid receipt PDFs on disk, clean ones and matched
tampers that each carry exactly one provenance defect:

* ``editor_tag``        — /Producer naming an image/PDF editor
* ``date_mismatch``     — ModDate well after CreationDate
* ``incremental_update``— a second body+xref appended (two ``%%EOF`` markers)

``build_pdf`` is a self-contained byte-layout writer (no third-party dep): all
object offsets and the xref are computed here, so the files are real PDFs the
inspector parses. Content is ASCII, so str length == latin-1 byte length and the
computed offsets are exact.
"""

from __future__ import annotations

import random
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

from ..models import DocumentType, FraudType, LabeledSample, Receipt
from .synth import Dataset

_LEGIT_PRODUCERS = [
    "ERPNext", "Tally.ERP 9", "SAP Crystal Reports", "QuickBooks",
    "Zoho Books", "Microsoft Print To PDF", "HP Smart Scan", "Canon CanoScan",
]
_EDITOR_PRODUCERS = [
    "Adobe Photoshop 24.1", "iLovePDF", "Smallpdf", "PDFescape",
    "Foxit PhantomPDF", "Sejda", "Canva",
]
_VENDORS = ["Reliance Fresh", "Croma", "Apollo Pharmacy", "Cafe Coffee Day", "Big Bazaar"]


def _pdf_date(dt: datetime) -> str:
    return "D:" + dt.strftime("%Y%m%d%H%M%S")


def build_pdf(info: dict[str, str], *, incremental: Optional[dict[str, str]] = None) -> bytes:
    """Lay out a one-page PDF whose Info dict is ``info``. If ``incremental`` is
    given, append a second revision rewriting the Info object (yields two
    ``%%EOF`` markers, i.e. one incremental update)."""

    def info_obj(d: dict[str, str]) -> str:
        return "<< " + " ".join(f"/{k} ({v})" for k, v in d.items()) + " >>"

    stream = "BT /F1 12 Tf 72 720 Td (Receipt) Tj ET"
    objs = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << >> >>"),
        (4, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"),
        (5, info_obj(info)),
    ]

    out = "%PDF-1.7\n"
    offsets: dict[int, int] = {}
    for num, body in objs:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{body}\nendobj\n"

    size = len(objs) + 1  # + free object 0
    startxref = len(out)
    out += f"xref\n0 {size}\n0000000000 65535 f \n"
    for num, _ in objs:
        out += f"{offsets[num]:010d} 00000 n \n"
    out += f"trailer\n<< /Size {size} /Root 1 0 R /Info 5 0 R >>\n"
    out += f"startxref\n{startxref}\n%%EOF\n"

    if incremental is not None:
        prev = startxref
        upd_off = len(out)
        out += f"5 0 obj\n{info_obj(incremental)}\nendobj\n"
        new_xref = len(out)
        out += f"xref\n0 1\n0000000000 65535 f \n5 1\n{upd_off:010d} 00000 n \n"
        out += f"trailer\n<< /Size {size} /Root 1 0 R /Info 5 0 R /Prev {prev} >>\n"
        out += f"startxref\n{new_xref}\n%%EOF\n"

    return out.encode("latin-1")


def _receipt(rng: random.Random, doc_id: str, path: Path, today: Date) -> Receipt:
    return Receipt(
        doc_id=doc_id,
        vendor_name=rng.choice(_VENDORS),
        date=today - timedelta(days=rng.randint(1, 180)),
        source=DocumentType.PDF,
        source_path=str(path),
    )


def _write(workdir: Path, doc_id: str, data: bytes) -> Path:
    path = workdir / f"{doc_id}.pdf"
    path.write_bytes(data)
    return path


def _issued_at(rng: random.Random, today: Date) -> datetime:
    day = today - timedelta(days=rng.randint(1, 180))
    return datetime(day.year, day.month, day.day, rng.randint(8, 20), rng.randint(0, 59))


def generate_pdf(
    n_clean: int = 40,
    fraud_per_type: int = 15,
    seed: int = 0,
    today: Optional[Date] = None,
    workdir: Union[str, Path, None] = None,
) -> Dataset:
    """Build a reproducible PDF provenance benchmark, writing the files under
    ``workdir`` and returning a :class:`Dataset` of labelled PDF samples."""

    rng = random.Random(seed)
    today = today or Date.today()
    workdir = Path(workdir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    samples: list[LabeledSample] = []
    counter = 0

    def make(is_fraud: bool, ftypes: set, data: bytes, detail: dict) -> None:
        nonlocal counter
        counter += 1
        doc_id = f"pdf-{counter:04d}"
        path = _write(workdir, doc_id, data)
        receipt = _receipt(rng, doc_id, path, today)
        samples.append(LabeledSample(receipt, is_fraud, ftypes, detail))

    for _ in range(n_clean):
        issued = _issued_at(rng, today)
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        make(False, {FraudType.NONE}, build_pdf(info), {})

    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        producer = rng.choice(_EDITOR_PRODUCERS)
        info = {"Producer": producer, "Creator": rng.choice(_LEGIT_PRODUCERS),
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        make(True, {FraudType.METADATA}, build_pdf(info), {"mode": "editor_tag", "producer": producer})

    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        modified = issued + timedelta(days=rng.randint(15, 400))
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(modified)}
        make(True, {FraudType.METADATA}, build_pdf(info),
             {"mode": "date_mismatch", "gap_days": (modified - issued).days})

    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        # append an incremental revision that changes nothing material -> isolates
        # the "edited after issuance" signal (one extra %%EOF), no editor/date flags
        data = build_pdf(info, incremental=dict(info))
        make(True, {FraudType.METADATA}, data, {"mode": "incremental_update"})

    rng.shuffle(samples)
    return Dataset(history=[], samples=samples)
