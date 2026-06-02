"""A learned, transparent fuser — logistic regression over the SAME per-detector
confidence-weighted signals the noisy-OR baseline consumes.

Why this exists: the false-positive audit showed that on real receipts the binding
constraint is a *noisy* arithmetic signal (it fires both on genuine fraud and on
lossy extractions of legitimate receipts), while the structural detectors (tax-id,
date, duplicate, provenance) are high-precision. Noisy-OR weights every detector
equally; a learned fuser can down-weight the unreliable signal and lean on the
reliable ones — the mechanism by which it can cut the real false-positive rate at a
matched fraud-recall. The model is a plain logistic regression so its per-detector
weights are directly inspectable (see :meth:`LearnedFuser.explain`).

Honesty: the positives it trains on are SYNTHETIC fraud and the negatives are real
*legitimate* receipts (plus synthetic clean ones). So it learns to separate
*synthetic fraud from real-legitimate receipts* — NOT to detect real fraud. It is a
strict superset of noisy-OR's inputs (one feature per detector = ``signal.weighted``),
so it can only do better on that separation if the equal-weight assumption was the
thing hurting us — which is exactly what the audit predicted. Noisy-OR stays the
zero-training default (:class:`slipguard.fusion.Fuser`); this is opt-in.

Dependency-free by design: a ~30-line full-batch gradient descent, deterministic
(zero init, no RNG), so a fit is reproducible from the same rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .detectors import default_detectors
from .models import Signal


def feature_names() -> list[str]:
    """The feature ordering: one slot per canonical detector, by name. Names (not
    positions) key the features, so a fuser fit on this order keeps working even if
    ``default_detectors()`` is later reordered — only adding/removing a detector
    changes the vector."""
    return [d.name for d in default_detectors()]


def feature_vector(signals: Sequence[Signal], order: Sequence[str]) -> list[float]:
    """Map a receipt's signals to a fixed feature vector: feature *i* = the
    confidence-weighted score (``score * confidence``) of the detector named
    ``order[i]``, or 0.0 if that detector did not report (or abstained — an
    abstainer has confidence 0, hence ``weighted == 0``). These are the exact
    quantities noisy-OR combines, so the logistic model is noisy-OR's inputs with
    learned per-detector weights instead of an implicit equal weight."""
    by_name = {s.detector: s for s in signals}
    return [by_name[name].weighted if name in by_name else 0.0 for name in order]


def _sigmoid(z: float) -> float:
    # Two-branch form avoids overflow in exp() for large-magnitude z.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    l2: float = 1e-3,
    lr: float = 0.5,
    iters: int = 5000,
    balance: bool = True,
) -> tuple[list[float], float]:
    """Fit ``P(fraud) = sigmoid(w·x + b)`` by full-batch gradient descent on the
    (optionally class-balanced) log-loss with L2 on the weights.

    ``balance=True`` reweights the two classes to equal total mass so the typically
    far-more-numerous legitimate receipts don't simply teach the model to always say
    "clean". Deterministic: weights start at zero and there is no sampling, so the
    same rows always yield the same fit."""
    n = len(X)
    if n == 0:
        raise ValueError("cannot fit on an empty training set")
    dim = len(X[0])

    if balance:
        n_pos = sum(1 for v in y if v >= 0.5)
        n_neg = n - n_pos
        w_pos = n / (2.0 * n_pos) if n_pos else 0.0
        w_neg = n / (2.0 * n_neg) if n_neg else 0.0
        sw = [w_pos if v >= 0.5 else w_neg for v in y]
    else:
        sw = [1.0] * n
    sw_total = sum(sw) or 1.0

    w = [0.0] * dim
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * dim
        gb = 0.0
        for xi, yi, wi in zip(X, y, sw):
            z = b + sum(w[j] * xi[j] for j in range(dim))
            err = (_sigmoid(z) - yi) * wi
            for j in range(dim):
                gw[j] += err * xi[j]
            gb += err
        for j in range(dim):
            # mean gradient + L2 shrinkage (bias is left unregularised)
            w[j] -= lr * (gw[j] / sw_total + l2 * w[j])
        b -= lr * (gb / sw_total)
    return w, b


@dataclass
class LearnedFuser:
    """A fitted logistic fuser. Callable as ``risk = fuser(signals)`` so it can be
    dropped straight into :class:`slipguard.fusion.Fuser` as its ``combiner``."""

    detector_order: list[str]
    weights: list[float]
    bias: float

    def __call__(self, signals: Sequence[Signal]) -> float:
        x = feature_vector(signals, self.detector_order)
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return _sigmoid(z)

    @classmethod
    def fit(
        cls,
        rows: Sequence[tuple[Sequence[Signal], bool]],
        order: Sequence[str] | None = None,
        **fit_kw,
    ) -> "LearnedFuser":
        """Fit from ``(signals, is_fraud)`` rows. ``order`` defaults to the canonical
        detector names."""
        order = list(order) if order is not None else feature_names()
        X = [feature_vector(sigs, order) for sigs, _ in rows]
        y = [1.0 if label else 0.0 for _, label in rows]
        w, b = fit_logistic(X, y, **fit_kw)
        return cls(order, w, b)

    def explain(self) -> dict[str, float]:
        """The learned per-detector weights (plus ``_bias``), sorted by magnitude —
        a positive weight means "this detector firing pushes toward fraud", and a
        small/negative one means the model found it unreliable. This is the whole
        point of choosing logistic regression: the fusion rule is legible."""
        items = sorted(
            zip(self.detector_order, self.weights), key=lambda kv: -abs(kv[1])
        )
        out = {name: round(w, 4) for name, w in items}
        out["_bias"] = round(self.bias, 4)
        return out
