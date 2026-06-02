from datetime import date

from slipguard.data.pdfsynth import build_pdf, build_signed_pdf, generate_pdf
from slipguard.detectors import default_detectors
from slipguard.detectors.pdfmeta import PdfMetadataDetector
from slipguard.eval.harness import evaluate
from slipguard.forensics.pdf import inspect_pdf
from slipguard.fusion import Fuser
from slipguard.models import DocumentType, Receipt

_CLEAN = {"Producer": "ERPNext", "Creator": "ERPNext",
          "CreationDate": "D:20260115120000", "ModDate": "D:20260115120000"}


def _pdf_receipt(tmp_path, name: str, data: bytes) -> Receipt:
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(data)
    return Receipt(name, "Croma", date(2026, 1, 10),
                   source=DocumentType.PDF, source_path=str(path))


# --- inspector ---------------------------------------------------------------

def test_inspect_clean_pdf():
    p = inspect_pdf(build_pdf(_CLEAN))
    assert p.eof_count == 1 and p.incremental_updates == 0
    assert p.producer == "ERPNext" and p.editor_tag is None
    assert p.date_gap_days == 0.0


def test_inspect_editor_tag():
    info = dict(_CLEAN, Producer="Adobe Photoshop 24.1")
    assert inspect_pdf(build_pdf(info)).editor_tag == "photoshop"


def test_inspect_date_mismatch():
    info = dict(_CLEAN, ModDate="D:20260320120000")
    assert inspect_pdf(build_pdf(info)).date_gap_days == 64.0


def test_inspect_incremental_update():
    p = inspect_pdf(build_pdf(_CLEAN, incremental=dict(_CLEAN)))
    assert p.eof_count == 2 and p.incremental_updates == 1


def test_inspect_content_edit_is_localized():
    # an incremental revision that rewrites the page CONTENT stream (object 4) is
    # localized as a content edit via the /Prev xref object-diff.
    p = inspect_pdf(build_pdf(_CLEAN, incremental_content="Total 900.00"))
    assert p.incremental_updates == 1 and p.content_stream_edits == 1


def test_inspect_metadata_incremental_is_not_a_content_edit():
    # rewriting only the Info dict (object 5) is an incremental update but NOT a content
    # edit — the diff distinguishes a metadata patch from a displayed-value patch.
    p = inspect_pdf(build_pdf(_CLEAN, incremental=dict(_CLEAN)))
    assert p.incremental_updates == 1 and p.content_stream_edits == 0


def test_inspect_clean_has_no_content_edit():
    assert inspect_pdf(build_pdf(_CLEAN)).content_stream_edits == 0


def test_inspect_clean_signed_pdf_is_fully_covered():
    p = inspect_pdf(build_signed_pdf(_CLEAN))
    assert p.is_signed and p.signature_uncovered_bytes == 0


def test_inspect_edit_after_signing_is_flagged():
    # bytes appended after the signature's /ByteRange -> edit-after-signing
    p = inspect_pdf(build_signed_pdf(_CLEAN, tamper=True))
    assert p.is_signed and p.signature_uncovered_bytes > 0


def test_inspect_malformed_never_raises():
    p = inspect_pdf(b"not a real pdf at all")
    assert p.eof_count == 0 and p.producer is None


# --- detector ----------------------------------------------------------------

def test_detector_clean_is_low(tmp_path):
    s = PdfMetadataDetector().score(_pdf_receipt(tmp_path, "c", build_pdf(_CLEAN)))
    assert not s.abstained and s.score < 0.1


def test_detector_editor_is_high(tmp_path):
    data = build_pdf(dict(_CLEAN, Producer="iLovePDF"))
    assert PdfMetadataDetector().score(_pdf_receipt(tmp_path, "e", data)).score > 0.6


def test_detector_date_mismatch_is_high(tmp_path):
    data = build_pdf(dict(_CLEAN, ModDate="D:20260601120000"))
    assert PdfMetadataDetector().score(_pdf_receipt(tmp_path, "d", data)).score > 0.6


def test_detector_incremental_is_high(tmp_path):
    data = build_pdf(_CLEAN, incremental=dict(_CLEAN))
    assert PdfMetadataDetector().score(_pdf_receipt(tmp_path, "i", data)).score > 0.6


def test_detector_content_edit_is_high_and_explained(tmp_path):
    # a content-stream rewrite scores high AND is explained + evidenced as such, not just
    # as a generic incremental update — the localization the /Prev diff buys.
    data = build_pdf(_CLEAN, incremental_content="Total 900.00")
    s = PdfMetadataDetector().score(_pdf_receipt(tmp_path, "ce", data))
    assert s.score > 0.6
    assert any("content stream" in r for r in s.reasons)
    assert s.evidence["content_stream_edits"] == 1


def test_detector_clean_signed_is_low(tmp_path):
    # a legitimately signed PDF (signature-only AcroForm, /ByteRange covers the file) must
    # NOT be flagged — neither for the signature nor as a fillable-form overlay.
    s = PdfMetadataDetector().score(_pdf_receipt(tmp_path, "sig-ok", build_signed_pdf(_CLEAN)))
    assert not s.abstained and s.score < 0.1
    assert s.evidence["is_signed"] is True and s.evidence["signature_uncovered_bytes"] == 0


def test_detector_edit_after_signing_is_high_and_explained(tmp_path):
    data = build_signed_pdf(_CLEAN, tamper=True)
    s = PdfMetadataDetector().score(_pdf_receipt(tmp_path, "sig-bad", data))
    assert s.score > 0.6
    assert any("signing" in r for r in s.reasons)
    assert s.evidence["signature_uncovered_bytes"] > 0


def test_detector_abstains_without_source_path():
    r = Receipt("x", "V", date(2026, 1, 1), source=DocumentType.PDF)
    assert PdfMetadataDetector().score(r).abstained


def test_detector_abstains_on_missing_file():
    r = Receipt("x", "V", date(2026, 1, 1),
                source=DocumentType.PDF, source_path="D:/nope/missing.pdf")
    assert PdfMetadataDetector().score(r).abstained


def test_detector_abstains_on_non_pdf_route():
    # run() (not score()) gates by document type
    r = Receipt("x", "V", date(2026, 1, 1), source=DocumentType.IMAGE)
    assert PdfMetadataDetector().run(r).abstained


# --- harness -----------------------------------------------------------------

def test_pdf_benchmark_is_strong(tmp_path):
    dataset = generate_pdf(seed=0, today=date(2026, 6, 1), workdir=tmp_path)
    report = evaluate(dataset, default_detectors(), Fuser())
    assert report.n_fraud > 0

    by_name = {d.name: d for d in report.detectors}
    assert by_name["pdf_meta"].target_recall > 0.9
    assert by_name["pdf_meta"].fp_rate < 0.1
    # structured detectors have nothing to read on a bare PDF -> they abstain
    for name in ("arithmetic", "tax_id", "duplicate"):
        assert by_name[name].n_target == 0

    assert report.fusion.auc > 0.95
    assert report.fusion.fp_rate < 0.1
