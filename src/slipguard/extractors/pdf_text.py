"""Born-digital PDF extraction via pypdfium2 — the PDF-route extractor.

Why this exists: the threat model covers born-digital PDFs as much as phone photos, but
until now the PDF route had only *provenance* forensics (``pdf_meta``) — nothing read its
fields, so the robust lead detectors (``arithmetic`` / ``tax_id`` / ``date_sanity``) never
ran on a PDF. This closes that gap: it reads the PDF's embedded text layer and hands the
positioned lines to the shared keyword/position KIE (:mod:`slipguard.extractors.kie`),
the *same* layer the docTR OCR extractor uses — so a PDF scores end-to-end exactly as an
image does, and the KIE is written once.

Licence (commercial-safe): pypdfium2 is Apache-2.0 / BSD-3-Clause and bundles Google's
PDFium (BSD-3-Clause); no copyleft, no AGPL (we deliberately avoid PyMuPDF's AGPL). The
import is **lazy inside** :meth:`extract`, so importing this module — and the whole
package — stays dependency-free; ``available()`` reports a missing dep via ``find_spec``
without importing anything.

Confidence: a born-digital PDF's text layer is *exact* (it is the characters the producer
wrote, not a recognition guess), so every line is read at confidence 1.0 and
``field_confidence`` stays empty (== fully trusted). That is honest — there is no
recognition uncertainty to report — but note it is a *reading* confidence, not a
field-*labelling* one (the KIE could still pick the wrong line as "total"; see
:mod:`~slipguard.extractors.kie`).

Honest limitation: this reads the embedded **text layer** only. A scanned-image PDF (a
photo wrapped in a PDF, no text layer) yields no text and an empty Receipt — that document
needs OCR and belongs on the IMAGE route (rasterise-then-OCR is future work, not done
here). Born-digital PDFs — the half of the threat model this route is for — do carry a
text layer.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from ..models import DocumentType, Receipt
from .base import Extractor
from .kie import Line, receipt_from_lines


def _lines_from_rects(
    rects: list[tuple[str, float, float, float, float]],
    page_width: float,
    page_height: float,
) -> list[Line]:
    """Convert pypdfium2 text rects ``(text, left, bottom, right, top)`` (PDF points,
    bottom-left origin) into KIE :class:`~slipguard.extractors.kie.Line`s. ``y`` is
    normalised to page-top = 0 (so the KIE's top-band / bottom-most comparisons work) and
    ``x`` to a [0,1] fraction; confidence is a constant 1.0 (digital text is exact). Pure,
    so the geometry mapping is unit-testable without opening a PDF.

    Lines are returned sorted top-to-bottom, as the KIE expects."""
    lines: list[Line] = []
    for text, left, _bottom, _right, top in rects:
        text = " ".join((text or "").split())  # collapse any internal newlines/runs
        if not text:
            continue
        y = 1.0 - (top / page_height) if page_height else 0.0
        x = left / page_width if page_width else 0.0
        lines.append(Line(text=text, y=y, conf=1.0, x=x))
    lines.sort(key=lambda ln: ln.y)
    return lines


class PdfTextExtractor(Extractor):
    """Read a born-digital PDF's embedded text layer (pypdfium2) and map it onto a Receipt
    via the shared KIE. Handles the PDF route so PDFs score end-to-end like images."""

    name = "pdf_text"
    handles = (DocumentType.PDF,)

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("pypdfium2") is None:
            return False, 'missing dep: pypdfium2 — pip install -e ".[pdf]"'
        return True, ""

    @staticmethod
    def _page_rects(textpage) -> list[tuple[str, float, float, float, float]]:
        """One ``(text, left, bottom, right, top)`` per rectangular text area on a page.
        Isolated so :meth:`extract` reads cleanly; the pure geometry mapping is
        :func:`_lines_from_rects`."""
        out: list[tuple[str, float, float, float, float]] = []
        for i in range(textpage.count_rects()):
            left, bottom, right, top = textpage.get_rect(i)
            text = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)
            out.append((text, left, bottom, right, top))
        return out

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        import pypdfium2 as pdfium

        lines: list[Line] = []
        pdf = pdfium.PdfDocument(path)
        try:
            for page_index, page in enumerate(pdf):
                width, height = page.get_size()
                textpage = page.get_textpage()
                try:
                    rects = self._page_rects(textpage)
                finally:
                    textpage.close()
                # Offset each later page below the previous one so "bottom-most total"
                # and top-band vendor still hold across a multi-page invoice (a 1-page
                # receipt keeps y in [0,1] unchanged).
                for ln in _lines_from_rects(rects, width, height):
                    ln.y += page_index
                    lines.append(ln)
                page.close()
        finally:
            pdf.close()

        return receipt_from_lines(
            lines, doc_id or path, source=DocumentType.PDF, source_path=path
        )
