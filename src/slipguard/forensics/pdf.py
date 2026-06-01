"""Dependency-free PDF provenance inspection.

The robust, hard-to-fake provenance signals for an edited receipt PDF are:

* **incremental updates** — a PDF written once has a single ``%%EOF``; every later
  in-place edit appends another body + xref + ``%%EOF``. Billing systems emit
  their PDF once, so >1 generation is a strong "edited after issuance" signal.
* **editor tags** — the ``/Producer`` / ``/Creator`` metadata naming an image or
  PDF *editor* (Photoshop, iLovePDF, …) rather than the issuing system or a scanner.
* **CreationDate vs ModDate** — a modification timestamp well after creation.

This first layer parses the raw bytes (regex over the literal Info dictionary).
It is intentionally light and license-clean; it does **not** yet decode
cross-reference streams or compressed/XMP metadata, so on those PDFs the string
fields may be ``None`` while the ``%%EOF`` count stays reliable. Production
hardening (pikepdf/pdfid for xref-stream + XMP) is tracked in the roadmap.
"""

from __future__ import annotations

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


def _match_editor(producer: Optional[str], creator: Optional[str]) -> Optional[str]:
    haystack = " ".join(v for v in (producer, creator) if v).lower()
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
