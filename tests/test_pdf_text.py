"""Tests for the born-digital PDF extractor's *own* parts — mapping pypdfium2's text
rects (PDF points, bottom-left origin) into the shared KIE's ``Line`` contract, and the
PDF-route binding — plus a real round-trip proving a minted text PDF reads back into the
right Receipt fields. The paradigm-agnostic KIE itself is tested in ``test_kie.py``.

The round-trip block is the PDF analogue of the docTR/VLM oracle: a receipt whose fields
are known is rendered to a real PDF, then read back through pypdfium2 + the KIE, and the
fields must survive — so a PDF now scores end-to-end exactly as an image does.
"""

from __future__ import annotations

import importlib.util
from datetime import date as Date

import pytest

from slipguard.data.pdfsynth import (
    build_text_pdf,
    generate_pdf_extraction,
    render_receipt_lines,
)
from slipguard.extractors import PdfTextExtractor, pdf_extractors
from slipguard.extractors.pdf_text import _lines_from_rects
from slipguard.models import DocumentType, LineItem, Receipt

_HAVE_PDFIUM = importlib.util.find_spec("pypdfium2") is not None
_needs_pdfium = pytest.mark.skipif(not _HAVE_PDFIUM, reason="pypdfium2 not installed ([pdf] extra)")


# --- text rects -> positioned lines (pypdfium2-specific) ---------------------

def test_lines_from_rects_maps_geometry():
    # one rect near the top of a 612x792 page: (text, left, bottom, right, top)
    (ln,) = _lines_from_rects([("STORE", 72.0, 700.0, 200.0, 720.0)], 612.0, 792.0)
    assert ln.text == "STORE"
    assert ln.conf == 1.0                       # born-digital text is exact
    assert abs(ln.y - (1 - 720 / 792)) < 1e-9   # page-top = 0 convention
    assert abs(ln.x - (72 / 612)) < 1e-9


def test_lines_from_rects_collapses_whitespace_and_drops_empty():
    rects = [
        ("  TOTAL\n  5.50 ", 72.0, 100.0, 300.0, 120.0),
        ("   ", 72.0, 80.0, 300.0, 90.0),        # blank -> dropped
    ]
    lines = _lines_from_rects(rects, 612.0, 792.0)
    assert [ln.text for ln in lines] == ["TOTAL 5.50"]


def test_lines_from_rects_sorts_top_to_bottom():
    # pass the bottom-of-page rect first; the top one (larger 'top') must come out first
    low = ("BOTTOM", 72.0, 90.0, 200.0, 110.0)   # top=110 -> y≈0.86
    high = ("TOP", 72.0, 700.0, 200.0, 720.0)    # top=720 -> y≈0.09
    assert [ln.text for ln in _lines_from_rects([low, high], 612.0, 792.0)] == ["TOP", "BOTTOM"]


def test_lines_from_rects_degenerate_page_is_defensive():
    (ln,) = _lines_from_rects([("X", 10.0, 0.0, 20.0, 10.0)], 0.0, 0.0)
    assert ln.x == 0.0 and ln.y == 0.0           # zero page size -> top-left, no crash


# --- extractor contract + registry ------------------------------------------

def test_pdf_text_handles_only_pdf_route():
    ex = PdfTextExtractor()
    assert ex.can_handle(DocumentType.PDF)
    assert not ex.can_handle(DocumentType.IMAGE)
    assert not ex.can_handle(DocumentType.STRUCTURED)


def test_pdf_text_available_returns_bool_reason_tuple():
    ok, reason = PdfTextExtractor().available()
    assert isinstance(ok, bool) and isinstance(reason, str)


def test_pdf_extractors_registry_handles_pdf():
    exs = pdf_extractors()
    assert any(isinstance(e, PdfTextExtractor) for e in exs)
    assert all(e.can_handle(DocumentType.PDF) for e in exs)


# --- synthetic render contract (no PDF opened) -------------------------------

def test_render_receipt_lines_shape():
    r = Receipt(
        doc_id="d", vendor_name="Croma", date=Date(2020, 1, 2),
        line_items=[LineItem("Coffee", 1.0, 3.50, 3.50)],
        subtotal=3.50, tax_amount=0.35, total=3.85,
        source=DocumentType.PDF, source_path="/x.pdf",
    )
    assert render_receipt_lines(r) == [
        "Croma", "2020-01-02", "Coffee  3.50",
        "Subtotal  3.50", "Tax  0.35", "Total  3.85",
    ]


# --- real round-trip: minted text PDF -> pypdfium2 + KIE -> Receipt ----------

@_needs_pdfium
def test_extract_reads_back_minted_fields(tmp_path):
    r = Receipt(
        doc_id="rt", vendor_name="Apollo Pharmacy", date=Date(2020, 3, 4),
        line_items=[LineItem("Notebook", 1.0, 12.00, 12.00),
                    LineItem("Pens", 1.0, 3.00, 3.00)],
        subtotal=15.00, tax_amount=1.50, total=16.50,
        source=DocumentType.PDF, source_path="/x.pdf",
    )
    path = tmp_path / "rt.pdf"
    path.write_bytes(build_text_pdf(render_receipt_lines(r), {"Producer": "ERPNext"}))

    out = PdfTextExtractor().extract(str(path), doc_id="rt")
    assert out.source is DocumentType.PDF       # PDF provenance route, not IMAGE
    assert out.source_path == str(path) and out.image_path is None
    assert out.subtotal == 15.00 and out.tax_amount == 1.50 and out.total == 16.50
    assert out.date == Date(2020, 3, 4)
    assert "Apollo" in (out.vendor_name or "")
    assert len(out.line_items) == 2
    assert out.field_confidence == {}           # exact text -> fully trusted, guard off


@_needs_pdfium
def test_generate_pdf_extraction_round_trips_on_oracle(tmp_path):
    # The numeric fields the robust detectors actually consume must survive the round
    # trip on every minted receipt (vendor is the one fragile field — short store names
    # can lose the top-band letter-count tie — so it is measured, not asserted here).
    truths = generate_pdf_extraction(n=6, seed=1, workdir=tmp_path)
    assert len(truths) == 6
    ex = PdfTextExtractor()
    for t in truths:
        out = ex.extract(t.source_path, doc_id=t.doc_id)
        assert out.subtotal == t.subtotal
        assert out.tax_amount == t.tax_amount
        assert out.total == t.total
        assert out.date == t.date
        assert len(out.line_items) == len(t.line_items)
