"""Generate fake / tampered receipts from real ones — the "fraud positives" the legitimate-only
corpora lack, for end-to-end testing of the detectors.

The pure-Python method (Pillow) overlays an **altered total** or a **future date** onto a real
receipt image. The result is deliberately BOTH a visible edit (a pasted-looking patch) and a
content break (the printed numbers/date no longer reconcile) — so a tampered slip is caught by the
*reliable* deterministic detectors (arithmetic / date_sanity) once the model reads the altered value,
not only by the unreliable pixel-forensics path.

No GPU, no API, no new dependency beyond Pillow — so this runs anywhere, including a work laptop.
(Two heavier methods — Gemini "Nano Banana" image edits and local diffusion — live behind the
`make-fakes` CLI's other ``--method`` values.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ._common import list_source_images


def _font(size: int):
    """A bold TrueType font if the OS has one, else Pillow's bitmap default."""
    from PIL import ImageFont
    for name in ("arialbd.ttf", "Arial Bold.ttf", "arial.ttf",
                 "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _overlay(img, text: str, where: str = "bottom"):
    """Paint ``text`` in a white box over the receipt (a crude 'sticker' edit). Returns a new RGB
    image. ``where`` is ``bottom`` (over the totals area) or ``top`` (over the header/date)."""
    from PIL import ImageDraw
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    size = max(16, w // 13)
    font = _font(size)
    try:
        tw = d.textlength(text, font=font)
    except Exception:
        tw = size * len(text) * 0.55
    pad = max(6, size // 3)
    x = max(pad, (w - tw) / 2)
    y = (h - size * 2) if where == "bottom" else int(h * 0.12)
    d.rectangle([x - pad, y - pad, x + tw + pad, y + size + pad], fill="white", outline="black")
    d.text((x, y), text, fill="black", font=font)
    return im


def inflate_total(img):
    """Paint an inflated grand total over a full-width strip at the bottom (where the real total
    usually sits, so it's covered) → the model reads $9,999.99, which no longer reconciles with
    the receipt's subtotal + tax, so the deterministic arithmetic check fires too."""
    from PIL import ImageDraw
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    size = max(18, w // 11)
    font = _font(size)
    text = "TOTAL DUE  $9,999.99"
    strip_top = h - int(size * 2.4)
    d.rectangle([0, strip_top, w, h], fill="white", outline="black")
    try:
        tw = d.textlength(text, font=font)
    except Exception:
        tw = size * len(text) * 0.55
    d.text((max(8, (w - tw) / 2), strip_top + size * 0.5), text, fill="black", font=font)
    return im


def future_date(img):
    """Stamp a future transaction date → trips date_sanity once read."""
    return _overlay(img, "Date: 2099-12-31", where="top")


#: tamper name -> function. Each produces one fake per source image.
TAMPERS: dict[str, Callable] = {
    "inflated_total": inflate_total,
    "future_date": future_date,
}


def make_pytamper(src_dir: str, out_dir: str, limit: Optional[int] = None) -> list[Path]:
    """Read receipts from ``src_dir`` and write each TAMPERS variant into ``out_dir`` as
    ``<name>__<tamper>.jpg``. Returns the written paths. Pillow only — no GPU/API."""
    from PIL import Image
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srcs = list_source_images(src, limit)
    made: list[Path] = []
    for p in srcs:
        try:
            im = Image.open(p)
        except Exception:
            continue
        try:  # close the source handle even if a tamper raises (fd/lock leak on Windows)
            for name, fn in TAMPERS.items():
                dst = out / f"{p.stem}__{name}.jpg"
                fn(im).save(dst, "JPEG", quality=90)
                made.append(dst)
        finally:
            im.close()
    return made
