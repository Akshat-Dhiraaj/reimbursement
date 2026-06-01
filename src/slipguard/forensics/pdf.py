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

    return PdfProvenance(
        eof_count=eof_count,
        producer=producer,
        creator=creator,
        creation_date=creation,
        mod_date=mod,
        editor_tag=_match_editor(producer, creator),
        date_gap_days=_date_gap_days(creation, mod),
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

    @property
    def has_structural_risk(self) -> bool:
        return (self.has_acroform or self.has_javascript
                or self.has_open_action or self.overlay_annotations > 0)


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
        has_acroform=N.AcroForm in pdf.Root,
        has_javascript=_has_javascript(pdf),
        has_open_action=N.OpenAction in pdf.Root,
        overlay_annotations=_count_overlay_annots(pdf),
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
