"""PDF provenance inspection — two layers, by design.

The robust, hard-to-fake provenance signals for an edited receipt PDF are:

* **incremental updates** — a PDF written once has a single ``%%EOF``; every later
  in-place edit appends another body + xref + ``%%EOF``. Billing systems emit
  their PDF once, so >1 generation is a strong "edited after issuance" signal.
* **editor tags** — the ``/Producer`` / ``/Creator`` (and XMP ``xmp:CreatorTool``)
  metadata naming an image or PDF *editor* (Photoshop, iLovePDF, …) rather than the
  issuing system or a scanner.
* **CreationDate vs ModDate** — a modification timestamp well after creation.
* **structural anomalies** — a receipt that is a fillable **AcroForm**, carries
  **JavaScript** / an auto-run **OpenAction**, or has **overlay annotations**
  (a FreeText/Redact/Stamp box that can cover the old value) is structurally unlike
  an issued, flat receipt.

**Layer 1 — :func:`inspect_pdf` (dependency-free).** Parses the raw bytes (regex over
the literal Info dictionary). Light and license-clean, and the ``%%EOF`` count is
always reliable. Its blind spot: a **modern compressed PDF** (PDF 1.5+ stores the
Info dict inside an xref/object stream, and metadata may live only in an XMP packet),
where the literal ``/Producer (...)`` never appears — so producer/creator/dates read
``None``. Most real ERP/portal exports are exactly such compressed PDFs.

**Layer 2 — :func:`inspect_pdf_deep` (pikepdf, the optional ``[pdf-forensics]`` extra).**
pikepdf (MPL-2.0, a qpdf binding; commercial-safe, deliberately not AGPL PyMuPDF)
decodes xref/object streams and XMP, so it **recovers the editor tag and date gap that
Layer 1 misses on compressed PDFs**, and surfaces the structural anomalies above. It is
optional — :func:`pikepdf_available` gates it, the import is lazy, and it returns
``None`` (never raises) when the dep is absent or the file is unreadable, so the
dependency-free core keeps working unchanged.
"""

from __future__ import annotations

import importlib.util
import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

#: lowercase substrings that, in /Producer or /Creator, indicate an editor was
#: used (as opposed to an issuing ERP/POS system or a scanner driver).
KNOWN_EDITORS = frozenset({
    "photoshop", "gimp", "inkscape", "illustrator", "canva",
    "ilovepdf", "smallpdf", "sejda", "pdfescape", "soda pdf",
    "nitro", "foxit phantompdf", "pdf editor", "pdf-xchange editor",
    "libreoffice draw", "pdffiller", "dochub",
})

_STR_FIELD = "/{key}\\s*\\(((?:[^()\\\\]|\\\\.)*)\\)"  # /Key (literal string)


@dataclass
class PdfProvenance:
    eof_count: int
    producer: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[str] = None
    mod_date: Optional[str] = None
    editor_tag: Optional[str] = None       # which editor matched, if any
    date_gap_days: Optional[float] = None  # mod_date - creation_date, in days
    #: incremental updates that rewrote a page /Contents stream (the *displayed* values),
    #: localized by diffing the /Prev xref chain. 0 for a clean file, a metadata-only
    #: incremental update, or a compressed PDF whose xref the byte scan can't read.
    content_stream_edits: int = 0
    is_signed: bool = False                 # carries a digital signature (a /ByteRange)
    #: bytes appended AFTER the signature's /ByteRange — an incremental update applied
    #: after signing (edit-after-signing). 0 when unsigned or the signature spans the
    #: whole file. /ByteRange must be in the clear (it addresses byte offsets), so this
    #: works even on compressed PDFs, dependency-free.
    signature_uncovered_bytes: int = 0

    @property
    def incremental_updates(self) -> int:
        """Number of edits appended after the original write (0 for a clean file)."""
        return max(0, self.eof_count - 1)


def inspect_pdf(data: Union[bytes, str, Path]) -> PdfProvenance:
    """Inspect a PDF given as raw bytes or a path. Never raises on malformed
    input — the worst case is ``None`` string fields with a valid ``eof_count``."""
    raw = data if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
    text = bytes(raw).decode("latin-1", "ignore")

    eof_count = text.count("%%EOF")
    producer = _last_field(text, "Producer")
    creator = _last_field(text, "Creator")
    creation = _last_field(text, "CreationDate")
    mod = _last_field(text, "ModDate")

    content_objs = _content_object_numbers(text)
    content_edits = sum(1 for changed in _incremental_changed_objects(text)
                        if changed & content_objs)
    # len(text) == byte length: latin-1 maps all 256 byte values 1:1, so "ignore" drops
    # nothing and char offsets equal byte offsets (what /ByteRange counts).
    is_signed, sig_uncovered = _signature_coverage(text, len(text))

    return PdfProvenance(
        eof_count=eof_count,
        producer=producer,
        creator=creator,
        creation_date=creation,
        mod_date=mod,
        editor_tag=_match_editor(producer, creator),
        date_gap_days=_date_gap_days(creation, mod),
        content_stream_edits=content_edits,
        is_signed=is_signed,
        signature_uncovered_bytes=sig_uncovered,
    )


def _last_field(text: str, key: str) -> Optional[str]:
    # last occurrence reflects the effective value after any incremental update
    matches = re.findall(_STR_FIELD.format(key=key), text)
    return matches[-1].strip() if matches else None


def _match_editor(*values: Optional[str]) -> Optional[str]:
    """First known-editor substring found across any of the supplied tool strings
    (``/Producer``, ``/Creator``, and — for the deep layer — XMP ``pdf:Producer`` /
    ``xmp:CreatorTool``). Matching one combined haystack keeps the rule in one place."""
    haystack = " ".join(v for v in values if v).lower()
    for editor in KNOWN_EDITORS:
        if editor in haystack:
            return editor
    return None


def _parse_pdf_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", value)
    if not m:
        return None
    y, mo, d, hh, mm, ss = (int(g) if g else 0 for g in m.groups())
    try:
        return datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None


def _date_gap_days(creation: Optional[str], mod: Optional[str]) -> Optional[float]:
    c, m = _parse_pdf_date(creation), _parse_pdf_date(mod)
    if c is None or m is None:
        return None
    return (m - c).total_seconds() / 86400.0


# --- /Prev incremental-update object diff (localizes *what* was edited) --------
#
# "an incremental update exists" (eof_count > 1) is coarse — legitimate incremental
# updates happen too (a digital signature, a form fill). The stronger, more specific
# signal is *which* object an update rewrote: if it rewrote a page /Contents stream, the
# displayed values were patched after issuance. We recover that by diffing the classic
# xref /Prev chain from raw bytes (dependency-free). Honest blind spot: a *compressed*
# PDF stores its cross-references in an xref stream the byte scan can't read, so no
# content edit is localized there (reported as 0, never a false positive).

def _content_object_numbers(text: str) -> set[int]:
    """Object numbers referenced as a page's ``/Contents`` — the streams that carry the
    *displayed* text/values. Handles the single-ref (``/Contents 4 0 R``) and array
    (``/Contents [4 0 R 6 0 R]``) forms."""
    nums: set[int] = set()
    for m in re.finditer(r"/Contents\s+(\d+)\s+0\s+R", text):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"/Contents\s*\[([^\]]*)\]", text):
        for ref in re.finditer(r"(\d+)\s+0\s+R", m.group(1)):
            nums.add(int(ref.group(1)))
    return nums


def _incremental_changed_objects(text: str) -> list[set[int]]:
    """For each incremental update (a classic ``xref`` section whose trailer carries
    ``/Prev``), the set of object numbers it (re)defines. Empty when the file is a single
    revision or uses cross-reference *streams* (compressed PDFs) the byte scan can't read."""
    changes: list[set[int]] = []
    # Non-greedy xref..trailer blocks; \bxref\b excludes the "xref" inside "startxref".
    for m in re.finditer(r"\bxref\b(.*?)\btrailer\b\s*(<<.*?>>)", text, re.DOTALL):
        body, trailer = m.group(1), m.group(2)
        if "/Prev" not in trailer:
            continue  # the base cross-reference section, not an incremental update
        objs = _objects_in_xref_body(body)
        if objs:
            changes.append(objs)
    return changes


def _objects_in_xref_body(body: str) -> set[int]:
    """Object numbers marked in-use (``n``) across the subsections of one classic xref
    table. Each subsection is a ``<first> <count>`` header followed by ``count`` entries
    (``<10-digit offset> <5-digit gen> <n|f>``); object 0 (the free-list head) is excluded."""
    objs: set[int] = set()
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        header = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", lines[i])
        if header:
            start, count = int(header.group(1)), int(header.group(2))
            # Clamp to the lines actually present so a malformed/hostile header like
            # "0 9999999999" can't spin a multi-billion-iteration loop (the module
            # promises never to hang on malformed input).
            real = min(count, max(0, len(lines) - (i + 1)))
            for k in range(real):
                entry = re.fullmatch(r"\s*(\d{10})\s+(\d{5})\s+([nf])\s*", lines[i + 1 + k])
                if entry and entry.group(3) == "n" and (start + k) != 0:
                    objs.add(start + k)
            i += real + 1
        else:
            i += 1
    return objs


def _signature_coverage(text: str, filesize: int) -> tuple[bool, int]:
    """Whether the PDF carries a digital signature, and how many trailing bytes fall
    OUTSIDE the signature's ``/ByteRange``. A PDF signature signs every byte except its
    own ``/Contents`` hex, so its ``/ByteRange`` must reach the end of the file; bytes
    beyond it are an incremental update appended *after* signing — an edit-after-signing
    tamper. With multiple signatures we take the furthest-reaching ``/ByteRange`` (the
    most recent signature); content past even that is unsigned. ``/ByteRange`` is always
    in the clear (it addresses raw offsets), so this needs no PDF parser."""
    ends = [int(g[2]) + int(g[3])
            for g in re.findall(r"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]", text)]
    if not ends:
        return False, 0
    return True, max(0, filesize - max(ends))


# --- Layer 2: pikepdf deep inspector (optional [pdf-forensics] extra) ---------

#: annotation subtypes that can visually COVER existing content — the "white box
#: over the old total, then retype it" tamper. Link/Widget/Popup are structural or
#: benign (form widgets, hyperlinks), so counting only these keeps the FP cost low.
_OVERLAY_ANNOTS = frozenset({"/FreeText", "/Redact", "/Stamp", "/Square", "/Highlight"})


def pikepdf_available() -> tuple[bool, str]:
    """Whether the deep layer can run, mirroring ``Extractor.available()``. Lets a
    caller (detector / benchmark) skip the deep inspection with a reason instead of
    crashing when the ``[pdf-forensics]`` extra isn't installed."""
    if importlib.util.find_spec("pikepdf") is None:
        return False, "pikepdf not installed ([pdf-forensics] extra)"
    return True, "ok"


@dataclass
class DeepPdfProvenance:
    """Provenance recovered by pikepdf that the byte-regex :class:`PdfProvenance`
    cannot see: metadata decoded through xref/object streams and XMP (so the editor
    tag and date gap survive on compressed PDFs), plus structural risk flags. Every
    field degrades to ``None`` / ``0`` / ``False`` rather than raising."""
    pdf_version: Optional[str] = None
    producer: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[str] = None
    mod_date: Optional[str] = None
    xmp_creator_tool: Optional[str] = None   # XMP xmp:CreatorTool — often the editor
    xmp_history_events: int = 0              # xmpMM:History edit-event count
    editor_tag: Optional[str] = None         # matched across docinfo + XMP tool strings
    date_gap_days: Optional[float] = None
    has_acroform: bool = False               # receipt rendered as a fillable form
    has_javascript: bool = False             # embedded JS — receipts never need it
    has_open_action: bool = False            # auto-run action on open
    overlay_annotations: int = 0             # cover-and-relabel annotation overlays
    content_overlays: int = 0                # cover-and-relabel drawn IN the content stream

    @property
    def has_structural_risk(self) -> bool:
        return (self.has_acroform or self.has_javascript or self.has_open_action
                or self.overlay_annotations > 0 or self.content_overlays > 0)


def inspect_pdf_deep(data: Union[bytes, str, Path]) -> Optional[DeepPdfProvenance]:
    """Deep provenance via pikepdf, or ``None`` if the extra is missing or the file
    won't open. Accepts raw bytes or a path (bytes are wrapped so tests can mint a
    fixture in memory). Never raises — every failure path returns ``None`` so the
    caller transparently falls back to the byte layer."""
    if not pikepdf_available()[0]:
        return None
    import pikepdf

    src: Union[str, io.BytesIO]
    src = io.BytesIO(bytes(data)) if isinstance(data, (bytes, bytearray)) else str(data)
    try:
        with pikepdf.open(src) as pdf:
            return _deep_from_pdf(pdf)
    except Exception:
        # pikepdf is stricter than the byte scanner; a parse failure must not crash
        # the detector — Layer 1 still produced a result.
        return None


def _deep_from_pdf(pdf) -> DeepPdfProvenance:
    import pikepdf
    N = pikepdf.Name

    di = pdf.docinfo  # decoded even when the Info dict lives in an object stream
    producer = _obj_str(di.get(N.Producer))
    creator = _obj_str(di.get(N.Creator))
    creation = _obj_str(di.get(N.CreationDate))
    mod = _obj_str(di.get(N.ModDate))

    xmp_tool = xmp_producer = None
    xmp_history = 0
    try:
        with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
            xmp_tool = _xmp_str(meta.get("xmp:CreatorTool"))
            xmp_producer = _xmp_str(meta.get("pdf:Producer"))
            hist = meta.get("xmpMM:History")
            xmp_history = len(hist) if isinstance(hist, list) else 0
    except Exception:
        pass  # no or malformed XMP packet — leave the XMP fields empty

    return DeepPdfProvenance(
        pdf_version=str(pdf.pdf_version),
        producer=producer, creator=creator, creation_date=creation, mod_date=mod,
        xmp_creator_tool=xmp_tool, xmp_history_events=xmp_history,
        editor_tag=_match_editor(producer, creator, xmp_producer, xmp_tool),
        date_gap_days=_date_gap_days(creation, mod),
        has_acroform=_has_fillable_acroform(pdf),
        has_javascript=_has_javascript(pdf),
        has_open_action=N.OpenAction in pdf.Root,
        overlay_annotations=_count_overlay_annots(pdf),
        content_overlays=_count_content_overlays(pdf),
    )


def _obj_str(value) -> Optional[str]:
    """A pikepdf String/Name -> stripped str, or None for missing/blank."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _xmp_str(value) -> Optional[str]:
    # XMP values can be a str or a list (dc:* are arrays); take the first non-empty.
    if isinstance(value, list):
        value = value[0] if value else None
    return _obj_str(value)


def _has_javascript(pdf) -> bool:
    """True if document-level JavaScript is present — either a ``/Names /JavaScript``
    name tree or an ``/OpenAction`` whose action is ``/S /JavaScript``."""
    import pikepdf
    N = pikepdf.Name
    names = pdf.Root.get(N.Names)
    if names is not None and N.JavaScript in names:
        return True
    oa = pdf.Root.get(N.OpenAction)
    return isinstance(oa, pikepdf.Dictionary) and oa.get(N.S) == N.JavaScript


def _has_fillable_acroform(pdf) -> bool:
    """True for a fillable-DATA AcroForm (an editable overlay — the tamper cousin), but
    NOT for a signature-only form: an AcroForm with ``/SigFlags`` whose every field is
    ``/FT /Sig`` is a legitimately signed PDF (the documented benign cousin), so we don't
    flag it — otherwise every digitally-signed invoice trips the overlay-form signal."""
    import pikepdf
    N = pikepdf.Name
    acro = pdf.Root.get(N.AcroForm)
    if acro is None:
        return False
    fields = acro.get(N.Fields)
    n = len(fields) if fields is not None else 0
    if N.SigFlags in acro and n > 0:
        try:
            if all(f.get(N.FT) == N.Sig for f in fields):
                return False  # signature form, not a data-entry overlay
        except Exception:
            pass
    return True


def _count_overlay_annots(pdf) -> int:
    import pikepdf
    N = pikepdf.Name
    n = 0
    for page in pdf.pages:
        annots = page.get(N.Annots)
        if annots is None:
            continue
        for a in annots:
            try:
                if str(a.get(N.Subtype)) in _OVERLAY_ANNOTS:
                    n += 1
            except Exception:
                continue
    return n


def _count_content_overlays(pdf) -> int:
    """Count cover-and-relabel overlays drawn IN a page's content stream: a white (or
    near-white) filled rectangle whose device bbox covers the position of text drawn
    EARLIER in the *same* stream — the "white box over the old total, then retype it"
    tamper done at the content-stream level (vs. as an annotation, which
    :func:`_count_overlay_annots` catches). Low FP by construction: a legitimate table-cell
    fill draws its rectangle *before* the cell text, so nothing is underneath; only a fill
    landing on *pre-existing* text is flagged."""
    import pikepdf
    n = 0
    for page in pdf.pages:
        try:
            n += _overlays_in_stream(pikepdf.parse_content_stream(page))
        except Exception:
            continue  # unparsable content stream -> contribute nothing, never crash
    return n


def _pdf_nums(operands) -> Optional[list]:
    """Operands as floats, or ``None`` if any is non-numeric (a ``TJ`` array, a name) — so
    numeric operators (``cm``/``Tm``/``re``/``rg``/…) parse without choking on the
    string/array operands of the text operators."""
    try:
        return [float(o) for o in operands]
    except Exception:
        return None


def _overlays_in_stream(instructions) -> int:
    """Minimal content-stream interpreter: tracks the CTM (``q``/``Q``/``cm``), the text
    origin (``BT``/``Tm``/``Td``/``Tj``) and white nonstroking fills (``rg``/``g``/``k``)
    on rectangles (``re`` then a fill op), and counts white rectangles whose device bbox
    covers a text origin emitted earlier in the stream. Exact for identity/translation
    CTMs (what real receipts use); best-effort under rotation/scale."""
    ident = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def mul(m, o):  # compose PDF matrices (m applied first, then o)
        a, b, c, d, e, f = m
        a2, b2, c2, d2, e2, f2 = o
        return (a * a2 + b * c2, a * b2 + b * d2, c * a2 + d * c2,
                c * b2 + d * d2, e * a2 + f * c2 + e2, e * b2 + f * d2 + f2)

    def apply(m, x, y):
        return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])

    ctm, tm = ident, ident
    stack: list = []
    fill_white = False
    rect = None
    text_pts: list = []   # device origins of text already drawn (earlier in the stream)
    overlays = 0

    for instr in instructions:
        op = str(instr.operator)
        if op == "q":
            stack.append((ctm, fill_white))
        elif op == "Q":
            ctm, fill_white = stack.pop() if stack else (ident, False)
        elif op == "cm":
            v = _pdf_nums(instr.operands)
            if v and len(v) == 6:
                ctm = mul(tuple(v), ctm)
        elif op == "BT":
            tm = ident
        elif op == "Tm":
            v = _pdf_nums(instr.operands)
            if v and len(v) == 6:
                tm = tuple(v)
        elif op in ("Td", "TD"):
            v = _pdf_nums(instr.operands)
            if v and len(v) == 2:
                tm = mul((1.0, 0.0, 0.0, 1.0, v[0], v[1]), tm)
        elif op in ("Tj", "TJ", "'", '"'):
            text_pts.append(apply(mul(tm, ctm), 0.0, 0.0))
        elif op == "rg":
            v = _pdf_nums(instr.operands)
            fill_white = bool(v) and len(v) == 3 and all(c >= 0.9 for c in v)
        elif op == "g":
            v = _pdf_nums(instr.operands)
            fill_white = bool(v) and len(v) == 1 and v[0] >= 0.9
        elif op == "k":
            v = _pdf_nums(instr.operands)
            fill_white = bool(v) and len(v) == 4 and all(c <= 0.1 for c in v)
        elif op == "re":
            v = _pdf_nums(instr.operands)
            rect = tuple(v) if v and len(v) == 4 else None
        elif op in ("f", "F", "b", "B", "f*", "b*", "B*"):
            if rect is not None and fill_white:
                x, y, w, h = rect
                corners = [apply(ctm, x, y), apply(ctm, x + w, y),
                           apply(ctm, x, y + h), apply(ctm, x + w, y + h)]
                xs, ys = [p[0] for p in corners], [p[1] for p in corners]
                bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
                if any(bx0 <= px <= bx1 and by0 <= py <= by1 for px, py in text_pts):
                    overlays += 1
            rect = None
        elif op in ("n", "S", "s"):
            rect = None  # path ended without our fill
    return overlays
