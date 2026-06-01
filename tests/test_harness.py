from slipguard.data.synth import generate
from slipguard.detectors import default_detectors
from slipguard.eval.harness import evaluate
from slipguard.fusion import Fuser


def test_benchmark_runs_and_fusion_is_strong():
    report = evaluate(generate(seed=0), default_detectors(), Fuser())
    assert report.n_samples > 0 and report.n_fraud > 0
    # On synthetic fraud that violates these exact rules, fusion should separate
    # cleanly. This validates the harness + deterministic layer, not real-world fraud.
    assert report.fusion.auc > 0.95
    assert report.fusion.fp_rate < 0.1

    by_name = {d.name: d for d in report.detectors}
    for name in ("arithmetic", "tax_id", "date_sanity", "duplicate"):
        assert by_name[name].target_recall > 0.9
        assert by_name[name].fp_rate < 0.1


def test_fusion_beats_best_single_detector_overall():
    report = evaluate(generate(seed=0), default_detectors(), Fuser())
    best_single = max(d.auc for d in report.detectors)
    assert report.fusion.auc > best_single
