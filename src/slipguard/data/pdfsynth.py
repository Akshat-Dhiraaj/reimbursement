"""Synthetic PDF benchmark for the provenance route.

Mints tiny but structurally valid receipt PDFs on disk, clean ones and matched
tampers that each carry exactly one provenance defect. Two writers, one per layer:

**Byte layer (dependency-free).** :func:`build_pdf` lays out an *uncompressed* PDF whose
Info dict is literal text, so the byte scanner reads it directly. :func:`generate_pdf`
mints the matched corpus:

* ``editor_tag``        — /Producer naming an image/PDF editor
* ``date_mismatch``     — ModDate well after CreationDate
* ``incremental_update``— a second body+xref appended (two ``%%EOF`` markers)

All object offsets and the xref are computed here (content is ASCII, so str length ==
latin-1 byte length and the offsets are exact) — real PDFs the inspector parses, no
third-party writer.

**Deep layer (pikepdf, the ``[pdf-forensics]`` extra).** :func:`build_compressed_pdf` mints
a *compressed* PDF 1.5+ whose Info dict and metadata live in object streams — invisible to
the byte scanner, the modern-ERP-export shape the byte writer cannot reproduce.
:func:`generate_pdf_deep` mints the matched corpus: ``editor_tag`` / ``date_mismatch`` as
byte-layer-blind metadata frauds (only the deep layer recovers them), plus the structural
anomalies ``javascript`` / ``open_action`` / ``acroform`` / ``overlay``.
"""

from __future__ import annotations

import io
import random
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

from ..forensics.pdf import pikepdf_available
from ..models import DocumentType, FraudType, LabeledSample, LineItem, Receipt
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
#: Purchasable item names for the extraction corpus. Deliberately none contain a KIE
#: summary keyword (subtotal/tax/total/cash/change/...), so every one is harvested as a
#: line item and none is mistaken for a summary row.
_ITEM_NAMES = ["Coffee", "Sandwich", "Notebook", "Pens", "Cable",
               "Stapler", "Folder", "Mouse", "Adapter", "Mug"]


def _pdf_date(dt: datetime) -> str:
    return "D:" + dt.strftime("%Y%m%d%H%M%S")


def _info_obj(d: dict[str, str]) -> str:
    return "<< " + " ".join(f"/{k} ({v})" for k, v in d.items()) + " >>"


def _assemble_pdf(objs: list[tuple[int, str]], *, info_num: int) -> tuple[str, int, int]:
    """Serialise consecutively-numbered objects (1..N, in order) into a single-revision
    PDF body with a computed xref table and a trailer pointing /Info at ``info_num``.
    Returns ``(text, size, startxref)`` so a caller can append an incremental revision.
    Content is ASCII, so ``len(str) == latin-1 byte length`` and every offset is exact —
    this is what lets us write real, inspector-parsable PDFs with no third-party writer."""
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
    out += f"trailer\n<< /Size {size} /Root 1 0 R /Info {info_num} 0 R >>\n"
    out += f"startxref\n{startxref}\n%%EOF\n"
    return out, size, startxref


def _content_obj(text: str = "Receipt") -> str:
    """Page content-stream object (object 4 in :func:`build_pdf`) that prints ``text``.
    Factored out so an incremental revision can rewrite it with a *different* value — the
    way a fraudster patches a displayed amount after issuance — using the same writer."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    return f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"


def _append_revision(out: str, *, obj_num: int, body: str, size: int, prev: int) -> str:
    """Append one incremental revision that rewrites a single object ``obj_num`` and
    chains to the previous xref at byte offset ``prev`` via ``/Prev``. Yields a second
    ``%%EOF`` and a one-object xref subsection naming exactly the changed object — the
    structure :func:`slipguard.forensics.pdf.inspect_pdf` diffs to tell *what* was edited
    after issuance (a metadata object vs. the page content stream)."""
    upd_off = len(out)
    out += f"{obj_num} 0 obj\n{body}\nendobj\n"
    new_xref = len(out)
    out += f"xref\n0 1\n0000000000 65535 f \n{obj_num} 1\n{upd_off:010d} 00000 n \n"
    out += f"trailer\n<< /Size {size} /Root 1 0 R /Info 5 0 R /Prev {prev} >>\n"
    out += f"startxref\n{new_xref}\n%%EOF\n"
    return out


def build_pdf(
    info: dict[str, str],
    *,
    incremental: Optional[dict[str, str]] = None,
    incremental_content: Optional[str] = None,
) -> bytes:
    """Lay out a minimal one-page PDF whose Info dict is ``info`` (the *provenance*
    benchmark only needs a valid file carrying metadata, not readable fields).

    At most one incremental revision is appended (each yields a second ``%%EOF``, i.e. one
    incremental update), and they differ by *what* they rewrite — the distinction the
    ``/Prev`` object-diff localizes:

    * ``incremental``         — rewrites the **Info** object (object 5): a metadata-only edit.
    * ``incremental_content`` — rewrites the **page content stream** (object 4) with the given
      text: a *displayed value* patched after issuance, the stronger tamper signal."""
    objs = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << >> >>"),
        (4, _content_obj()),
        (5, _info_obj(info)),
    ]
    out, size, startxref = _assemble_pdf(objs, info_num=5)

    if incremental is not None:
        out = _append_revision(out, obj_num=5, body=_info_obj(incremental),
                               size=size, prev=startxref)
    if incremental_content is not None:
        out = _append_revision(out, obj_num=4, body=_content_obj(incremental_content),
                               size=size, prev=startxref)

    return out.encode("latin-1")


#: fixed-width fake signature hex. The forensic signal is whether /ByteRange COVERS the
#: file, not the cryptography, so the contents are just a stable-length placeholder.
_SIG_PLACEHOLDER = "0" * 512


def build_signed_pdf(info: dict[str, str], *, tamper: bool = False) -> bytes:
    """Mint a structurally *signed* one-page PDF: an AcroForm signature field whose Sig
    dict carries a ``/ByteRange`` patched to span the whole file around its ``/Contents``
    placeholder (a legitimately signed, untampered document). With ``tamper=True`` an
    incremental revision is appended AFTER the signed range, so those bytes fall outside
    ``/ByteRange`` — the edit-after-signing tell :func:`slipguard.forensics.pdf.inspect_pdf`
    reports as ``signature_uncovered_bytes``. We don't compute a real signature; the signal
    is byte-range coverage, not crypto validity."""
    sig = ("<< /Type /Sig /Filter /Adobe.PPKLite /SubFilter /adbe.pkcs7.detached "
           "/ByteRange [0 0000000000 0000000000 0000000000] "
           f"/Contents <{_SIG_PLACEHOLDER}> >>")
    objs = [
        (1, "<< /Type /Catalog /Pages 2 0 R /AcroForm 6 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << >> /Annots [7 0 R] >>"),
        (4, _content_obj()),
        (5, _info_obj(info)),
        (6, "<< /Fields [7 0 R] /SigFlags 3 >>"),
        (7, "<< /FT /Sig /T (Signature1) /Subtype /Widget /Rect [0 0 0 0] /P 3 0 R /V 8 0 R >>"),
        (8, sig),
    ]
    out, size, startxref = _assemble_pdf(objs, info_num=5)

    # Patch /ByteRange to bracket the /Contents placeholder and reach EOF. lt/gt are
    # computed before the same-length patch, so no offsets shift.
    lt = out.index("/Contents <") + len("/Contents ")
    gt = out.index(">", lt) + 1
    byte_range = f"[0 {lt:010d} {gt:010d} {len(out) - gt:010d}]"
    out = out.replace("[0 0000000000 0000000000 0000000000]", byte_range, 1)

    if tamper:
        # append a metadata-rewrite revision after the signed range -> bytes outside
        # /ByteRange (the signature no longer covers the whole file).
        out = _append_revision(out, obj_num=5, body=_info_obj(dict(info)),
                               size=size, prev=startxref)
    return out.encode("latin-1")


def build_overlay_pdf(info: dict[str, str]) -> bytes:
    """A text PDF whose content stream draws original text, then a WHITE rectangle over a
    row, then a relabelled value on top — the cover-and-relabel tamper done *in the content
    stream* (vs. as an annotation). The deep layer's content-overlay interpreter
    (:func:`slipguard.forensics.pdf._count_content_overlays`) should flag it; a clean text
    PDF (:func:`build_text_pdf`) draws no rectangle over pre-existing text and does not."""
    stream = (
        "BT /F1 11 Tf\n"
        "1 0 0 1 72 700 Tm (Reliance Fresh) Tj\n"
        "1 0 0 1 72 600 Tm (Total 100.00) Tj\n"
        "ET\n"
        "1 1 1 rg\n"                # white fill
        "70 595 120 16 re f\n"      # box over the "Total 100.00" row
        "0 0 0 rg\n"
        "BT /F1 11 Tf 1 0 0 1 74 598 Tm (Total 900.00) Tj ET"  # relabel on top
    )
    objs = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
        (4, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"),
        (5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        (6, _info_obj(info)),
    ]
    out, _size, _startxref = _assemble_pdf(objs, info_num=6)
    return out.encode("latin-1")


def _content_stream(body_lines: list[str]) -> str:
    """A text content stream that prints each of ``body_lines`` on its own row, evenly
    spread from near the top (y=750) to the bottom margin (y=80). Spreading over the full
    page height (rather than bunching at the top) keeps the KIE's top-band vendor pick and
    bottom-most total pick meaningful on a short receipt. Parens/backslashes are escaped
    per PDF string syntax (backslash first, so the escapes we add aren't re-escaped)."""
    rows = [ln for ln in body_lines if ln]
    n = len(rows)
    top, bottom = 750.0, 80.0
    out = ["BT", "/F1 11 Tf"]
    for i, text in enumerate(rows):
        y = top if n <= 1 else top - i * (top - bottom) / (n - 1)
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        out.append(f"1 0 0 1 72 {y:.1f} Tm ({esc}) Tj")
    out.append("ET")
    return "\n".join(out)


def build_text_pdf(body_lines: list[str], info: dict[str, str]) -> bytes:
    """Lay out a one-page PDF that actually *renders* ``body_lines`` as a readable text
    layer, using a standard-14 Helvetica font resource so pypdfium2 can extract the text
    back out. Same self-contained byte writer as :func:`build_pdf` (shared
    :func:`_assemble_pdf`), plus the /Font resource and a multi-line content stream — this
    is the corpus the PDF *extraction* benchmark reads (vs. the metadata-only provenance
    corpus from :func:`build_pdf`)."""
    stream = _content_stream(body_lines)
    objs = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
        (4, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"),
        (5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        (6, _info_obj(info)),
    ]
    out, _size, _startxref = _assemble_pdf(objs, info_num=6)
    return out.encode("latin-1")


def build_compressed_pdf(
    info: dict[str, str],
    *,
    javascript: bool = False,
    open_action: bool = False,
    acroform: bool = False,
    overlay: bool = False,
) -> bytes:
    """Mint a *compressed* one-page PDF with pikepdf — the corpus that exercises the deep
    layer. Saved with object streams (PDF 1.5+), so the Info dict lives inside an
    xref/object stream and the dependency-free byte scanner reads producer/creator/dates
    as ``None``: exactly the modern-ERP/portal-export blind spot the byte writers
    (:func:`build_pdf`) cannot reproduce. The optional flags inject the object-graph
    anomalies the deep layer surfaces, each independent of the others:

    * ``javascript``  — a document-level ``/Names /JavaScript`` name tree
    * ``open_action`` — an auto-run ``/OpenAction`` (a benign ``/Named`` action, so it is
      *not* also flagged as JavaScript)
    * ``acroform``    — a fillable ``/AcroForm``
    * ``overlay``     — a ``/FreeText`` annotation that can cover an existing value

    Requires the ``[pdf-forensics]`` extra; the import is lazy so this module still loads
    without it (callers gate on :func:`pikepdf_available`)."""
    import pikepdf
    from pikepdf import Array, Dictionary, Name, String

    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    for key, value in info.items():
        pdf.docinfo[Name("/" + key)] = String(value)

    if javascript:
        action = pdf.make_indirect(Dictionary(S=Name.JavaScript, JS=String("app.alert('x');")))
        pdf.Root.Names = Dictionary(JavaScript=Dictionary(Names=Array([String("EmbeddedJS"), action])))
    if open_action:
        pdf.Root.OpenAction = Dictionary(S=Name.Named, N=Name.NextPage)
    if acroform:
        pdf.Root.AcroForm = Dictionary(Fields=Array([]), NeedAppearances=True)
    if overlay:
        annot = pdf.make_indirect(Dictionary(
            Type=Name.Annot, Subtype=Name.FreeText,
            Rect=Array([100, 700, 300, 720]), Contents=String("100.00")))
        page.Annots = Array([annot])

    buf = io.BytesIO()
    pdf.save(buf, object_stream_mode=pikepdf.ObjectStreamMode.generate, compress_streams=True)
    return buf.getvalue()


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

    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        # incremental revision that rewrites the PAGE CONTENT stream (a displayed value),
        # not the Info dict -> the /Prev object-diff localizes a *content* edit, which is
        # stronger and more specific than a metadata-only incremental update (legit
        # incremental updates like signatures don't rewrite the page content).
        data = build_pdf(info, incremental_content="Total 900.00")
        make(True, {FraudType.METADATA}, data, {"mode": "content_edit"})

    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        # a digitally-signed PDF with content appended AFTER the signature's /ByteRange
        # (edit-after-signing) — the signature no longer covers the whole file.
        make(True, {FraudType.METADATA}, build_signed_pdf(info, tamper=True),
             {"mode": "signature_tamper"})

    rng.shuffle(samples)
    return Dataset(history=[], samples=samples)


def generate_pdf_deep(
    n_clean: int = 20,
    fraud_per_type: int = 10,
    seed: int = 0,
    today: Optional[Date] = None,
    workdir: Union[str, Path, None] = None,
) -> Dataset:
    """Build a *compressed*-PDF benchmark that exercises the pikepdf deep layer.

    Every file is a modern PDF 1.5+ written through pikepdf with object streams (see
    :func:`build_compressed_pdf`), so the Info dict and metadata are invisible to the
    dependency-free byte scanner — the real-ERP-export shape the uncompressed byte writer
    in :func:`generate_pdf` cannot represent. The fraud modes split into two groups that
    motivate the extra:

    * **metadata, byte-layer-blind** (``editor_tag``, ``date_mismatch``) — the defect lives
      in compressed metadata, so Layer 1 reads nothing and *only* the deep layer recovers
      it. This is the headline recall the extra buys; the ``eval-pdf-forensics`` CLI
      contrasts byte-only vs deep recall on exactly these.
    * **structural** (``javascript``, ``open_action``, ``acroform``, ``overlay``) — anomalies
      that exist only in the object graph, invisible to a byte scan by construction. Minted
      with a clean producer and aligned dates so the structure is the *only* fraud signal.

    Mirrors :func:`generate_pdf`'s ``make`` pattern. Requires the ``[pdf-forensics]`` extra;
    raises ``RuntimeError`` if pikepdf is absent so a caller never silently benchmarks an
    empty deep layer."""
    ok, why = pikepdf_available()
    if not ok:
        raise RuntimeError(f"generate_pdf_deep requires the [pdf-forensics] extra: {why}")

    rng = random.Random(seed)
    today = today or Date.today()
    workdir = Path(workdir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    samples: list[LabeledSample] = []
    counter = 0

    def make(is_fraud: bool, ftypes: set, data: bytes, detail: dict) -> None:
        nonlocal counter
        counter += 1
        doc_id = f"pdf-deep-{counter:04d}"
        path = _write(workdir, doc_id, data)
        receipt = _receipt(rng, doc_id, path, today)
        samples.append(LabeledSample(receipt, is_fraud, ftypes, detail))

    for _ in range(n_clean):
        issued = _issued_at(rng, today)
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        make(False, {FraudType.NONE}, build_compressed_pdf(info), {})

    # editor tag hidden in compressed metadata — Layer 1 blind, deep recovers it
    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        producer = rng.choice(_EDITOR_PRODUCERS)
        info = {"Producer": producer, "Creator": rng.choice(_LEGIT_PRODUCERS),
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        make(True, {FraudType.METADATA}, build_compressed_pdf(info),
             {"mode": "editor_tag", "producer": producer, "compressed": True})

    # mod-date well after creation, hidden in compressed metadata — Layer 1 blind
    for _ in range(fraud_per_type):
        issued = _issued_at(rng, today)
        modified = issued + timedelta(days=rng.randint(15, 400))
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(modified)}
        make(True, {FraudType.METADATA}, build_compressed_pdf(info),
             {"mode": "date_mismatch", "gap_days": (modified - issued).days, "compressed": True})

    # structural anomalies — clean producer/dates so the structure is the only signal
    for flag in ("javascript", "open_action", "acroform", "overlay"):
        for _ in range(fraud_per_type):
            issued = _issued_at(rng, today)
            producer = rng.choice(_LEGIT_PRODUCERS)
            info = {"Producer": producer, "Creator": producer,
                    "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
            make(True, {FraudType.METADATA}, build_compressed_pdf(info, **{flag: True}),
                 {"mode": flag})

    rng.shuffle(samples)
    return Dataset(history=[], samples=samples)


# --- born-digital PDF *extraction* corpus (readable fields) ------------------

def render_receipt_lines(receipt: Receipt) -> list[str]:
    """The text rows to print for ``receipt`` — vendor at the top, the ISO date, each
    line item as ``"<desc>  <amount>"``, then the subtotal/tax/total summary at the foot:
    the same shape a real receipt prints, so the shared KIE reads back exactly the fields
    the extraction benchmark scores."""
    rows: list[str] = []
    if receipt.vendor_name:
        rows.append(receipt.vendor_name)
    if receipt.date is not None:
        rows.append(receipt.date.isoformat())
    for it in receipt.line_items:
        rows.append(f"{it.description}  {it.amount:.2f}")
    if receipt.subtotal is not None:
        rows.append(f"Subtotal  {receipt.subtotal:.2f}")
    if receipt.tax_amount is not None:
        rows.append(f"Tax  {receipt.tax_amount:.2f}")
    if receipt.total is not None:
        rows.append(f"Total  {receipt.total:.2f}")
    return rows


def generate_pdf_extraction(
    n: int = 40,
    seed: int = 0,
    today: Optional[Date] = None,
    workdir: Union[str, Path, None] = None,
) -> list[Receipt]:
    """Mint ``n`` born-digital receipt PDFs whose fields are actually rendered as a text
    layer, returning the ground-truth Receipts (with ``source_path`` set to each written
    file). This is the oracle for the PDF *extraction* benchmark: ``PdfTextExtractor`` is
    run on each file and its output scored field-by-field against the returned truth,
    exactly as the IMAGE route is scored against WildReceipt's oracle.

    Each receipt is internally consistent (``subtotal == sum(items)``,
    ``total == subtotal + tax``), so it is a valid clean receipt and the field the KIE
    reads is the field we wrote — the benchmark measures *reading* fidelity, not arithmetic."""
    rng = random.Random(seed)
    today = today or Date.today()
    workdir = Path(workdir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    truths: list[Receipt] = []
    for i in range(1, n + 1):
        doc_id = f"pdf-ext-{i:04d}"
        path = workdir / f"{doc_id}.pdf"

        items: list[LineItem] = []
        for name in rng.sample(_ITEM_NAMES, rng.randint(1, 4)):
            amt = round(rng.uniform(1.0, 50.0), 2)
            items.append(LineItem(description=name, quantity=1.0, unit_price=amt, amount=amt))
        subtotal = round(sum(it.amount for it in items), 2)
        tax = round(subtotal * rng.choice([0.05, 0.08, 0.10, 0.18]), 2)
        total = round(subtotal + tax, 2)

        receipt = Receipt(
            doc_id=doc_id,
            vendor_name=rng.choice(_VENDORS),
            date=today - timedelta(days=rng.randint(1, 180)),
            currency="USD",
            country="US",
            line_items=items,
            subtotal=subtotal,
            tax_amount=tax,
            total=total,
            source=DocumentType.PDF,
            source_path=str(path),
        )

        issued = _issued_at(rng, today)
        producer = rng.choice(_LEGIT_PRODUCERS)
        info = {"Producer": producer, "Creator": producer,
                "CreationDate": _pdf_date(issued), "ModDate": _pdf_date(issued)}
        path.write_bytes(build_text_pdf(render_receipt_lines(receipt), info))
        truths.append(receipt)

    return truths
