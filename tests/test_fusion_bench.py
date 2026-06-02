"""Smoke test for the fusion comparison harness (eval/fusion_bench.py).

Runs the real synthetic pipeline (detectors + fit + scoring) but with NO real
corpora, so it's fast and dependency-free. It pins the contract — the learned fuser
separates synthetic fraud from synthetic clean, the real columns degrade to n/a when
no corpus is supplied, and the structured-only data leaves the provenance detectors
flagged inactive — without asserting brittle exact AUC values.
"""

from __future__ import annotations

import math

import pytest

from slipguard.data.synth import Dataset, generate
from slipguard.eval.fusion_bench import compare_fusion


def test_compare_fusion_runs_synthetic_only():
    train = generate(n_clean=20, fraud_per_type=6, n_history=20, seed=0)
    test = generate(n_clean=20, fraud_per_type=6, n_history=20, seed=1)

    cmp = compare_fusion(train, test, [], iters=800)  # no real corpora

    # learned fuser still separates synthetic fraud from synthetic clean
    assert cmp.synth_auc_learned > 0.9
    # with no real corpus, the real-FP / real-AUC columns are n/a (NaN)
    assert cmp.n_test_real == 0
    assert math.isnan(cmp.real_auc_learned)
    assert all(math.isnan(p.learned_fp) for p in cmp.points)
    # all six detectors are weighted; the two provenance ones never fire on structured data
    assert set(cmp.weights) == {
        "arithmetic", "tax_id", "date_sanity", "duplicate", "pdf_meta", "image_meta", "_bias",
    }
    assert "pdf_meta" in cmp.inactive and "image_meta" in cmp.inactive
    # report renders without error
    assert "Learned fusion vs noisy-OR" in str(cmp)


def test_multiroute_training_earns_provenance_weights(tmp_path):
    # the route-specific gap (#77): with PDF + image provenance fraud merged into training,
    # pdf_meta/image_meta earn non-zero weights instead of the structured-only zero above.
    pytest.importorskip("PIL")  # image synth (EXIF JPEGs) needs Pillow
    from slipguard.data.imagesynth import generate_image
    from slipguard.data.pdfsynth import generate_pdf

    def merged(seed: int) -> Dataset:
        s = generate(n_clean=20, fraud_per_type=6, n_history=20, seed=seed)
        pdf = generate_pdf(n_clean=10, fraud_per_type=5, seed=seed, workdir=tmp_path / f"pdf{seed}")
        img = generate_image(n_clean=10, fraud_per_type=5, seed=seed, workdir=tmp_path / f"img{seed}")
        return Dataset(history=s.history, samples=s.samples + pdf.samples + img.samples)

    cmp = compare_fusion(merged(0), merged(1), [], iters=800)
    assert "pdf_meta" not in cmp.inactive and "image_meta" not in cmp.inactive
    assert cmp.weights["pdf_meta"] > 0.0 and cmp.weights["image_meta"] > 0.0
