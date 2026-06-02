"""Tests for the C2PA / Content Credentials provenance signal (#76).

The manifest-classification logic is pure and unit-tested against representative C2PA
JSON. The detector's C2PA branch is tested by injecting a fake provenance, so it runs
without the heavy ``[c2pa]`` extra installed. One integration test, skipped unless c2pa +
Pillow are present, confirms the real ``Reader`` treats a plain JPEG (no manifest) as a
non-signal.

Why no end-to-end *signed* fixture: c2pa-rs enforces a strict signing-cert profile that
makes minting a self-signed test asset impractical, and C2PA detection is deterministic
schema-parsing (not a learned / AUC signal), so the parser is validated directly instead.
"""

from __future__ import annotations

from datetime import date

import pytest

from slipguard.detectors.imagemeta import ImageMetadataDetector
from slipguard.forensics.c2pa import C2paProvenance, classify_source_types
from slipguard.models import DocumentType, Receipt

_AI_URI = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
_CAM_URI = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"


def _manifest(source_uri: str) -> dict:
    return {"assertions": [{"label": "c2pa.actions",
                            "data": {"actions": [{"action": "c2pa.created",
                                                  "digitalSourceType": source_uri}]}}]}


# --- pure classification -----------------------------------------------------

def test_classify_ai_source():
    kind, uris = classify_source_types(_manifest(_AI_URI))
    assert kind == "ai" and uris == [_AI_URI]


def test_classify_camera_source():
    assert classify_source_types(_manifest(_CAM_URI))[0] == "camera"


def test_classify_unknown_when_no_source_type():
    kind, uris = classify_source_types({"assertions": [{"label": "c2pa.hash.data"}]})
    assert kind == "unknown" and uris == []


def test_classify_searches_nested_ingredients():
    # digitalSourceType can be nested inside an ingredient's manifest -> still found
    store = {"manifests": {"x": {"ingredients": [{"manifest": _manifest(_AI_URI)}]}}}
    assert classify_source_types(store)[0] == "ai"


def test_classify_ai_wins_over_camera():
    both = {"a": _manifest(_AI_URI), "b": _manifest(_CAM_URI)}
    assert classify_source_types(both)[0] == "ai"


# --- detector branch (inject provenance; no [c2pa] extra needed) --------------

def _img_receipt(tmp_path, name="x"):
    p = tmp_path / f"{name}.bin"
    p.write_bytes(b"not an image")  # EXIF read fails -> only the C2PA branch contributes
    return Receipt(name, "Croma", date(2026, 1, 10),
                   source=DocumentType.IMAGE, source_path=str(p))


def test_detector_flags_c2pa_ai(tmp_path, monkeypatch):
    monkeypatch.setattr("slipguard.detectors.imagemeta.c2pa_available", lambda: True)
    monkeypatch.setattr("slipguard.detectors.imagemeta.inspect_c2pa",
                        lambda path: C2paProvenance(True, "ai", (_AI_URI,), "Adobe Firefly"))
    s = ImageMetadataDetector().score(_img_receipt(tmp_path))
    assert not s.abstained and s.score > 0.85
    assert any("AI-generated" in r for r in s.reasons)
    assert s.evidence["c2pa_source_type"] == "ai"


def test_detector_camera_capture_weakly_exonerates(tmp_path, monkeypatch):
    monkeypatch.setattr("slipguard.detectors.imagemeta.c2pa_available", lambda: True)
    monkeypatch.setattr("slipguard.detectors.imagemeta.inspect_c2pa",
                        lambda path: C2paProvenance(True, "camera", (_CAM_URI,), "Pixel 10"))
    s = ImageMetadataDetector().score(_img_receipt(tmp_path))
    assert not s.abstained and s.score < 0.1
    assert any("camera capture" in r for r in s.reasons)


def test_detector_no_manifest_no_exif_abstains(tmp_path, monkeypatch):
    monkeypatch.setattr("slipguard.detectors.imagemeta.c2pa_available", lambda: True)
    monkeypatch.setattr("slipguard.detectors.imagemeta.inspect_c2pa",
                        lambda path: C2paProvenance(False))
    assert ImageMetadataDetector().score(_img_receipt(tmp_path)).abstained


# --- real Reader integration (needs the [c2pa] extra + Pillow) ---------------

def test_inspect_c2pa_abstains_on_plain_jpeg(tmp_path):
    pytest.importorskip("c2pa")
    pytest.importorskip("PIL")
    from PIL import Image

    from slipguard.forensics.c2pa import inspect_c2pa
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (16, 16), "white").save(p, "JPEG")
    assert inspect_c2pa(str(p)).has_manifest is False
