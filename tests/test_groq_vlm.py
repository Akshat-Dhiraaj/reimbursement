"""Tests for the Groq hosted-VLM extractor (#79).

Offline only: the ``extract()`` path makes a network request to Groq (data egress), so we
don't call it in CI. We pin the contract — it gates on ``GROQ_API_KEY``, labels itself
per-model, resolves via the ``groq`` spec, and base64-encodes an image to a data URL —
without a live request. The live field-accuracy number is produced by hand with
``slipguard eval-extract --extractor groq`` (see DECISIONS/ROADMAP)."""

from __future__ import annotations

import pytest

from slipguard.extractors import image_extractor_for_spec
from slipguard.extractors.groq_vlm import GroqVLExtractor, _data_url
from slipguard.models import DocumentType


def test_available_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ok, why = GroqVLExtractor().available()
    assert not ok and "GROQ_API_KEY" in why


def test_available_with_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert GroqVLExtractor().available()[0]


def test_name_and_route():
    ex = GroqVLExtractor()
    assert ex.name.startswith("groq:") and ex.can_handle(DocumentType.IMAGE)


def test_spec_resolves_to_groq():
    assert isinstance(image_extractor_for_spec("groq"), GroqVLExtractor)
    assert image_extractor_for_spec("groq:some/model").model_id == "some/model"


def test_data_url_encodes_image(tmp_path):
    pytest.importorskip("PIL")  # downscale path uses Pillow
    from PIL import Image
    p = tmp_path / "r.jpg"
    Image.new("RGB", (20, 20), "white").save(p, "JPEG")
    url = _data_url(str(p))
    assert url.startswith("data:image/jpeg;base64,") and len(url) > 40
