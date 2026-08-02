"""Prediction accuracy metrics (RMSE, MAE)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error, ignoring NaN predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    y_true, y_pred = y_true[mask], y_pred[mask]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error, ignoring NaN predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    y_true, y_pred = y_true[mask], y_pred[mask]
    return float(np.mean(np.abs(y_true - y_pred)))
