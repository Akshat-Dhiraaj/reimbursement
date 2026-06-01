from datetime import date

import pytest

pytest.importorskip("PIL")  # EXIF read/write needs Pillow (the [vlm] extra)
from PIL import Image

from slipguard.data.imagesynth import generate_image
from slipguard.detectors import default_detectors
from slipguard.detectors.imagemeta import ImageMetadataDetector
from slipguard.eval.harness import evaluate
from slipguard.forensics.image import inspect_image
from slipguard.fusion import Fuser
from slipguard.models import DocumentType, Receipt

_CAPTURE = "2019:06:01 09:00:00"


def _mint(tmp_path, name, *, software=None, captured=_CAPTURE, modified=_CAPTURE,
          make="Apple", model="iPhone 12", with_exif=True):
    """Mint a real JPEG independently of imagesynth, so the inspector is tested against
    images we built by hand (not by the module under test's own helper)."""
    path = tmp_path / f"{name}.jpg"
    img = Image.new("RGB", (32, 32), "white")
    if not with_exif:
        img.save(path, "JPEG")
        return path
    exif = Image.Exif()
    exif[0x0132] = modified           # DateTime
    if make:
        exif[0x010F] = make           # Make
    if model:
        exif[0x0110] = model          # Model
    if software:
        exif[0x0131] = software       # Software
    exif.get_ifd(0x8769)[0x9003] = captured  # DateTimeOriginal in the Exif sub-IFD
    img.save(path, "JPEG", exif=exif)
    return path


def _img_receipt(path, name):
    return Receipt(name, "Croma", date(2026, 1, 10),
                   source=DocumentType.IMAGE, source_path=str(path))


# --- inspector ---------------------------------------------------------------

def test_inspect_clean_image(tmp_path):
    p = inspect_image(str(_mint(tmp_path, "c")))
    assert p.has_exif and p.has_camera
    assert p.editor_tag is None and p.date_gap_days == 0.0


def test_inspect_editor_tag(tmp_path):
    p = inspect_image(str(_mint(tmp_path, "e", software="Adobe Photoshop 24.1")))
    assert p.editor_tag == "photoshop"


def test_inspect_date_mismatch(tmp_path):
    p = inspect_image(str(_mint(tmp_path, "d", modified="2019:08:04 09:00:00")))
    assert p.date_gap_days == 64.0


def test_inspect_no_exif_is_blank(tmp_path):
    p = inspect_image(str(_mint(tmp_path, "n", with_exif=False)))
    assert not p.has_exif and p.editor_tag is None


# --- detector ----------------------------------------------------------------

def test_detector_clean_is_low(tmp_path):
    s = ImageMetadataDetector().score(_img_receipt(_mint(tmp_path, "c"), "c"))
    assert not s.abstained and s.score < 0.1


def test_detector_editor_is_high(tmp_path):
    p = _mint(tmp_path, "e", software="GIMP 2.10.34")
    assert ImageMetadataDetector().score(_img_receipt(p, "e")).score > 0.6


def test_detector_date_mismatch_is_high(tmp_path):
    p = _mint(tmp_path, "d", modified="2019:08:04 09:00:00")
    assert ImageMetadataDetector().score(_img_receipt(p, "d")).score > 0.6


def test_detector_abstains_without_source_path():
    r = Receipt("x", "V", date(2026, 1, 1), source=DocumentType.IMAGE)
    assert ImageMetadataDetector().score(r).abstained


def test_detector_abstains_on_no_exif(tmp_path):
    # stripped/screenshot/AI images carry no EXIF -> abstain, never accuse
    r = _img_receipt(_mint(tmp_path, "n", with_exif=False), "n")
    assert ImageMetadataDetector().score(r).abstained


def test_detector_abstains_on_missing_file():
    r = Receipt("x", "V", date(2026, 1, 1),
                source=DocumentType.IMAGE, source_path="D:/nope/missing.jpg")
    assert ImageMetadataDetector().score(r).abstained


def test_detector_abstains_on_non_image_route():
    # run() (not score()) gates by document type
    r = Receipt("x", "V", date(2026, 1, 1), source=DocumentType.PDF)
    assert ImageMetadataDetector().run(r).abstained


# --- harness -----------------------------------------------------------------

def test_image_benchmark_is_strong(tmp_path):
    dataset = generate_image(seed=0, today=date(2026, 6, 1), workdir=tmp_path)
    report = evaluate(dataset, default_detectors(), Fuser())
    assert report.n_fraud > 0

    by_name = {d.name: d for d in report.detectors}
    assert by_name["image_meta"].target_recall > 0.9
    assert by_name["image_meta"].fp_rate < 0.1
    # structured detectors have nothing to read on a bare image -> they abstain
    for name in ("arithmetic", "tax_id", "duplicate"):
        assert by_name[name].n_target == 0

    assert report.fusion.auc > 0.95
    assert report.fusion.fp_rate < 0.1
