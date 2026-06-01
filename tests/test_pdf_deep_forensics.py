"""Deep PDF forensics (the pikepdf Layer 2). Skipped wholesale when the
[pdf-forensics] extra is absent, so the dependency-free suite stays green.

The through-line: every file here is a *compressed* PDF (object streams), so the
byte scanner is blind by construction and only the deep layer recovers the signal."""

from datetime import date

import pytest

from slipguard.data.pdfsynth import build_compressed_pdf, generate_pdf_deep
from slipguard.detectors import default_detectors
from slipguard.detectors.pdfmeta import PdfMetadataDetector
from slipguard.eval.harness import evaluate
from slipguard.forensics.pdf import inspect_pdf, inspect_pdf_deep, pikepdf_available
from slipguard.fusion import Fuser
from slipguard.models import DocumentType, Receipt

pytestmark = pytest.mark.skipif(
    not pikepdf_available()[0],
    reason="requires the [pdf-forensics] extra (pikepdf)",
)

_CLEAN = {"Producer": "ERPNext", "Creator": "ERPNext",
          "CreationDate": "D:20260115120000", "ModDate": "D:20260115120000"}


def _pdf_receipt(tmp_path, name: str, data: bytes) -> Receipt:
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(data)
    return Receipt(name, "Croma", date(2026, 1, 10),
                   source=DocumentType.PDF, source_path=str(path))


def _with_pdf_meta(use_deep: bool) -> list:
    """The default set with pdf_meta pinned to byte-only or deep — the same swap the
    eval-pdf-forensics CLI uses, so the contrast is scored on one shared corpus."""
    return [PdfMetadataDetector(use_deep=use_deep) if d.name == "pdf_meta" else d
            for d in default_detectors()]


# --- the compressed-PDF blind spot: byte layer blind, deep layer recovers ----

def test_compressed_hides_metadata_from_byte_layer():
    # an editor producer IS present, but lives in a compressed object stream
    data = build_compressed_pdf(dict(_CLEAN, Producer="iLovePDF", Creator="iLovePDF"))
    byte = inspect_pdf(data)
    assert byte.eof_count == 1 and byte.producer is None and byte.editor_tag is None
    deep = inspect_pdf_deep(data)
    assert deep.producer == "iLovePDF" and deep.editor_tag == "ilovepdf"


def test_deep_recovers_date_gap_on_compressed():
    data = build_compressed_pdf(dict(_CLEAN, ModDate="D:20260320120000"))
    assert inspect_pdf(data).date_gap_days is None        # byte layer blind
    assert inspect_pdf_deep(data).date_gap_days == 64.0   # deep recovers


def test_deep_clean_compressed_has_no_risk():
    deep = inspect_pdf_deep(build_compressed_pdf(_CLEAN))
    assert deep.editor_tag is None and deep.date_gap_days == 0.0
    assert not deep.has_structural_risk


# --- structural anomalies (deep layer only) ----------------------------------

def test_deep_flags_javascript():
    deep = inspect_pdf_deep(build_compressed_pdf(_CLEAN, javascript=True))
    assert deep.has_javascript and deep.has_structural_risk


def test_deep_open_action_is_independent_of_javascript():
    # a /Named OpenAction auto-runs on open but is NOT a JavaScript action
    deep = inspect_pdf_deep(build_compressed_pdf(_CLEAN, open_action=True))
    assert deep.has_open_action and not deep.has_javascript


def test_deep_flags_acroform():
    assert inspect_pdf_deep(build_compressed_pdf(_CLEAN, acroform=True)).has_acroform


def test_deep_flags_overlay_annotation():
    deep = inspect_pdf_deep(build_compressed_pdf(_CLEAN, overlay=True))
    assert deep.overlay_annotations == 1 and deep.has_structural_risk


def test_inspect_deep_never_raises_on_garbage():
    # pikepdf is stricter than the byte scan; a parse failure must degrade to None
    assert inspect_pdf_deep(b"not a real pdf at all") is None


# --- detector: the use_deep knob (deep on vs off) ----------------------------

def test_detector_blind_without_deep_on_compressed(tmp_path):
    # byte-only path: an editor fraud hidden in compressed metadata reads as clean
    data = build_compressed_pdf(dict(_CLEAN, Producer="iLovePDF"))
    s = PdfMetadataDetector(use_deep=False).score(_pdf_receipt(tmp_path, "b", data))
    assert not s.abstained and s.score < 0.1


def test_detector_recovers_editor_with_deep(tmp_path):
    data = build_compressed_pdf(dict(_CLEAN, Producer="iLovePDF"))
    s = PdfMetadataDetector(use_deep=True).score(_pdf_receipt(tmp_path, "e", data))
    assert s.score > 0.6


def test_detector_flags_javascript(tmp_path):
    data = build_compressed_pdf(_CLEAN, javascript=True)
    s = PdfMetadataDetector().score(_pdf_receipt(tmp_path, "j", data))
    assert s.score > 0.6


def test_detector_acroform_is_review_weight(tmp_path):
    # a fillable form alone is suspicious-not-damning: scored, but below the 0.5 auto-flag
    data = build_compressed_pdf(_CLEAN, acroform=True)
    s = PdfMetadataDetector().score(_pdf_receipt(tmp_path, "a", data))
    assert 0.3 < s.score < 0.5


# --- harness: the byte-only vs deep contrast as a regression -----------------

def test_deep_benchmark_contrast(tmp_path):
    dataset = generate_pdf_deep(seed=0, today=date(2026, 6, 1), workdir=tmp_path)
    assert dataset.samples  # minted

    deep = evaluate(dataset, _with_pdf_meta(use_deep=True), Fuser())
    byte = evaluate(dataset, _with_pdf_meta(use_deep=False), Fuser())

    deep_pm = {d.name: d for d in deep.detectors}["pdf_meta"]
    byte_pm = {d.name: d for d in byte.detectors}["pdf_meta"]
    assert byte_pm.target_recall == 0.0   # Layer 1 blind on compressed metadata + structure
    assert deep_pm.target_recall >= 0.8   # Layer 2 recovers it
    assert deep_pm.fp_rate == 0.0
    assert deep.fusion.recall == 1.0      # every fraud at least routed to human review
