"""Tests for the ExpressExpense loader (data/expressexpense.py).

ExpressExpense ships images only (no labels), so the loader just discovers image files and
emits image_path-only Receipts for the re-extraction FP audit. These use tiny placeholder
files (the loader globs by suffix; it never opens them), so no real images / network needed.
"""

from __future__ import annotations

import pytest

from slipguard.data.expressexpense import load_receipts
from slipguard.models import DocumentType


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_globs_images_recursively_sorted_and_skips_non_images(tmp_path):
    _touch(tmp_path / "b.jpg")
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "sub" / "c.jpeg")
    _touch(tmp_path / "readme.txt")        # not an image -> ignored
    _touch(tmp_path / "labels.json")       # not an image -> ignored

    receipts = load_receipts(tmp_path)
    names = [r.image_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for r in receipts]
    assert names == ["a.png", "b.jpg", "c.jpeg"]  # sorted, recursive, images only


def test_receipts_are_image_only_unlabelled():
    # one shared assertion: no oracle fields, IMAGE route, so oracle detectors abstain.
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    _touch(d / "r.jpg")
    r = load_receipts(d)[0]
    assert r.source is DocumentType.IMAGE
    assert r.image_path.endswith("r.jpg")
    assert r.vendor_name == "(unknown)" and r.date is None
    assert r.subtotal is None and r.total is None and r.line_items == []


def test_limit_caps_count(tmp_path):
    for i in range(5):
        _touch(tmp_path / f"{i}.jpg")
    assert len(load_receipts(tmp_path, limit=3)) == 3


def test_missing_dir_raises_with_fetch_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        load_receipts(tmp_path / "does-not-exist")
    assert "expressexpense" in str(ei.value).lower()  # carries the fetch URL/instructions
