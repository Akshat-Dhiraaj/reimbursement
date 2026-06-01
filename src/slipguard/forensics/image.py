"""Image EXIF provenance inspection — the metadata sibling of :mod:`slipguard.forensics.pdf`.

For a receipt *photo* the hard-to-fake "edited after capture" provenance signals are:

* **editor tag** — the EXIF ``Software`` field naming a raster/photo *editor*
  (Photoshop, GIMP, Snapseed, …) rather than a camera/phone pipeline. A genuine
  phone photo is written by the camera app, not by an image editor.
* **capture vs modify** — ``DateTimeOriginal`` (when the shot was taken) well
  before ``DateTime`` (when the file was last written): a re-save after editing.
  This mirrors the PDF inspector's CreationDate-vs-ModDate gap.

What we deliberately do **not** treat as guilt: **missing EXIF**. AI-generated
images and screenshots usually carry no EXIF — but so do perfectly legitimate
receipts shared through chat apps or downloaded from a portal, which strip it.
Absent EXIF therefore yields ``has_exif=False`` and the detector *abstains*; it is
not evidence either way. EXIF is also trivially strippable/forgeable, so a clean
read is only weak exoneration (the detector reflects that in its confidence).

Pillow is an optional (``[vlm]``) dependency, so it is imported lazily inside
:func:`inspect_image`; :func:`pillow_available` lets the detector check for it
*without* importing it and abstain cleanly when it is absent.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# EXIF tag ids we read (avoids a Pillow import just to name them).
_SOFTWARE = 0x0131          # base IFD: writing application
_DATETIME = 0x0132          # base IFD: file last-modified time
_MAKE = 0x010F              # base IFD: camera maker
_MODEL = 0x0110             # base IFD: camera model
_EXIF_IFD = 0x8769          # pointer to the Exif sub-IFD
_DATETIME_ORIGINAL = 0x9003  # Exif sub-IFD: capture time

#: lowercase substrings in EXIF ``Software`` that indicate a raster/photo *editor*
#: (as opposed to a camera/phone capture pipeline). Distinct from the PDF editor
#: list — these are image editors, that one names PDF tools.
KNOWN_IMAGE_EDITORS = frozenset({
    "photoshop", "lightroom", "gimp", "pixelmator", "paint.net",
    "affinity photo", "picsart", "snapseed", "canva", "fotor",
    "facetune", "photoscape", "befunky", "photo editor", "photopea",
})


@dataclass
class ImageProvenance:
    has_exif: bool
    software: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    datetime_original: Optional[str] = None  # capture time, as stored
    datetime_modified: Optional[str] = None  # last-write time, as stored
    editor_tag: Optional[str] = None          # which editor matched in Software, if any
    date_gap_days: Optional[float] = None      # modified - original, in days

    @property
    def has_camera(self) -> bool:
        """A camera make/model is present — corroborates a genuine capture."""
        return bool(self.make or self.model)


def pillow_available() -> bool:
    """Whether Pillow can be imported here, checked without importing it."""
    return importlib.util.find_spec("PIL") is not None


def inspect_image(path: str) -> ImageProvenance:
    """Inspect an image's EXIF provenance. Never raises on a non-image or EXIF-less
    file (returns ``has_exif=False``); a missing/unreadable path raises ``OSError``
    for the caller to handle. Requires Pillow (gate with :func:`pillow_available`)."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except UnidentifiedImageError:  # present but not a decodable image -> nothing to read
        return ImageProvenance(has_exif=False)

    if not len(exif):
        return ImageProvenance(has_exif=False)

    try:
        sub = exif.get_ifd(_EXIF_IFD)
    except Exception:  # malformed sub-IFD pointer -> treat as absent, keep base tags
        sub = {}

    software = _clean(exif.get(_SOFTWARE))
    original = _clean(sub.get(_DATETIME_ORIGINAL))
    modified = _clean(exif.get(_DATETIME))
    return ImageProvenance(
        has_exif=True,
        software=software,
        make=_clean(exif.get(_MAKE)),
        model=_clean(exif.get(_MODEL)),
        datetime_original=original,
        datetime_modified=modified,
        editor_tag=_match_editor(software),
        date_gap_days=_date_gap_days(original, modified),
    )


def _clean(value: object) -> Optional[str]:
    """EXIF strings can carry a trailing NUL or padding; normalise to a clean str."""
    if value is None:
        return None
    text = str(value).replace("\x00", "").strip()
    return text or None


def _match_editor(software: Optional[str]) -> Optional[str]:
    if not software:
        return None
    low = software.lower()
    for editor in KNOWN_IMAGE_EDITORS:
        if editor in low:
            return editor
    return None


def _parse_exif_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:  # canonical EXIF datetime: "YYYY:MM:DD HH:MM:SS"
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _date_gap_days(original: Optional[str], modified: Optional[str]) -> Optional[float]:
    o, m = _parse_exif_dt(original), _parse_exif_dt(modified)
    if o is None or m is None:
        return None
    return (m - o).total_seconds() / 86400.0
