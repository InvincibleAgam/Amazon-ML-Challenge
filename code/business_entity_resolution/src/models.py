"""Gradient-boosted matcher: LightGBM (MIT licence), trained from scratch.

Both stages use the same helpers; only the hyper-parameters differ.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from common import N_JOBS

PARAMS = {
    "stage1": dict(n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=200,
                   reg_lambda=1.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1),
    "stage2": dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, min_child_samples=100,
                   reg_lambda=1.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1),
    # second member of the stage-2 ensemble: deeper, slower, more randomised
    "stage2b": dict(n_estimators=1500, learning_rate=0.03, num_leaves=255, min_child_samples=50,
                    reg_lambda=3.0, colsample_bytree=0.6, subsample=0.7, subsample_freq=1,
                    random_state=7),
}


def to_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature columns -> float32 matrix (nulls become NaN, handled natively by LightGBM)."""
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def fit(X: np.ndarray, y: np.ndarray, stage: str, sample_weight=None,
        init_model: lgb.LGBMClassifier | None = None, seed_offset: int = 0) -> lgb.LGBMClassifier:
    """With `init_model` the fit continues that model for another n_estimators trees; the
    result holds the old and the new trees.  `seed_offset` gives them fresh row / column samples."""
    params = {"random_state": 0, **PARAMS[stage]}
    params["random_state"] += seed_offset
    m = lgb.LGBMClassifier(**params, n_jobs=N_JOBS, verbose=-1)
    m.fit(X, y, sample_weight=sample_weight,
          init_model=None if init_model is None else init_model.booster_)
    return m


def predict(models, X: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Average probability of one or several models, computed in chunks."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    flat = []                                   # a fold may itself hold an ensemble (list)
    for m in models:
        flat.extend(m if isinstance(m, (list, tuple)) else [m])
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        xs = X[i:i + chunk]
        out[i:i + chunk] = np.mean([m.predict_proba(xs)[:, 1] for m in flat], axis=0)
    return out


def predict_oof(models, X: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold: rows of fold f are scored by the model trained on the other fold."""
    out = np.empty(len(X), dtype=np.float32)
    for f in (0, 1):
        idx = np.flatnonzero(fold == f)
        if len(idx):
            out[idx] = predict(models[1 - f], X[idx])
    return out
