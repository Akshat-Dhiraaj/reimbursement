"""Offline tests for the fake-receipt generators (data/tamper.py + data/tamper_ai.py).

The pure-Python tamperer is exercised on tiny in-memory images; the AI methods are checked only for
their response-parsing and their fail-fast behaviour (no network)."""

import base64

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from slipguard.data import tamper, tamper_ai  # noqa: E402


def _make_src(tmp_path, n=2):
    d = tmp_path / "src"
    d.mkdir()
    for i in range(n):
        Image.new("RGB", (220, 320), "white").save(d / f"r{i}.jpg", "JPEG")
    return d


def test_pytamper_writes_two_fakes_per_image(tmp_path):
    made = tamper.make_pytamper(str(_make_src(tmp_path, 2)), str(tmp_path / "out"))
    assert len(made) == 4                                    # 2 images x 2 tamper types
    assert all(p.exists() for p in made)
    names = " ".join(p.name for p in made)
    assert "inflated_total" in names and "future_date" in names
    assert Image.open(made[0]).size == (220, 320)           # still a valid, same-size image


def test_pytamper_respects_limit(tmp_path):
    made = tamper.make_pytamper(str(_make_src(tmp_path, 5)), str(tmp_path / "out"), limit=2)
    assert len(made) == 4                                    # 2 sources (capped) x 2 tampers


def test_image_from_response_parses_inline_data():
    raw = b"\x89PNG\r\n\x1a\nFAKE"
    resp = {"candidates": [{"content": {"parts": [
        {"text": "here"},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(raw).decode()}}]}}]}
    assert tamper_ai._image_from_response(resp) == raw
    assert tamper_ai._image_from_response({"candidates": []}) is None   # text-only → None


def test_make_local_is_a_clear_not_setup_error(tmp_path):
    with pytest.raises(SystemExit):                          # heavy GPU setup — refuses, doesn't crash
        tamper_ai.make_local(str(tmp_path), str(tmp_path / "o"))
