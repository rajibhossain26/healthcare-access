"""Distance-decay functions.

Every function takes an array of travel costs (minutes, km, ...) and returns
weights of the same shape. Weights are 0 where the cost is outside the
catchment (or infinite / NaN), so the same functions work for both steps of
the floating catchment area (FCA) family of models.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

DecayFunction = Callable[[np.ndarray], np.ndarray]


def _as_float(cost) -> np.ndarray:
    arr = np.asarray(cost, dtype=float)
    return np.where(np.isnan(arr), np.inf, arr)


def binary(threshold: float) -> DecayFunction:
    """Weight 1 inside the catchment, 0 outside (classic 2SFCA)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    def f(cost):
        c = _as_float(cost)
        return (c <= threshold).astype(float)

    return f


def stepwise(breaks: Sequence[float], weights: Sequence[float]) -> DecayFunction:
    """Piecewise-constant weights (Enhanced 2SFCA, Luo & Qi 2009).

    ``breaks`` are the upper edges of each zone, e.g. ``[10, 20, 30]`` minutes,
    and ``weights`` the weight inside each zone, e.g. ``[1.0, 0.68, 0.22]``.
    Costs beyond the last break get weight 0.
    """
    breaks = np.asarray(breaks, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if breaks.ndim != 1 or len(breaks) != len(weights) or len(breaks) == 0:
        raise ValueError("breaks and weights must be 1-D and the same length")
    if np.any(np.diff(breaks) <= 0) or breaks[0] <= 0:
        raise ValueError("breaks must be positive and strictly increasing")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    def f(cost):
        c = _as_float(cost)
        idx = np.searchsorted(breaks, c, side="left")  # zone index; len(breaks) = outside
        lookup = np.append(weights, 0.0)
        return lookup[idx]

    return f


def gaussian(threshold: float) -> DecayFunction:
    """Gaussian decay truncated at the catchment edge (Dai 2010).

    w(d) = (exp(-0.5 (d/d0)^2) - exp(-0.5)) / (1 - exp(-0.5)) for d <= d0,
    which falls smoothly from 1 at d = 0 to 0 at d = d0.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    edge = np.exp(-0.5)

    def f(cost):
        c = _as_float(cost)
        with np.errstate(invalid="ignore", over="ignore"):
            w = (np.exp(-0.5 * (c / threshold) ** 2) - edge) / (1.0 - edge)
        return np.where(c <= threshold, w, 0.0)

    return f


def power(beta: float, min_cost: float = 1.0, threshold: float = np.inf) -> DecayFunction:
    """Inverse-power decay d**-beta used by the gravity model.

    Costs below ``min_cost`` are clamped to it so a provider at distance 0 does
    not produce an infinite weight.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if min_cost <= 0:
        raise ValueError("min_cost must be positive")

    def f(cost):
        c = _as_float(cost)
        clamped = np.maximum(c, min_cost)
        with np.errstate(divide="ignore", over="ignore"):
            w = clamped ** (-beta)
        return np.where(c <= threshold, w, 0.0)

    return f


def exponential(beta: float, threshold: float = np.inf) -> DecayFunction:
    """Negative-exponential decay exp(-beta * d)."""
    if beta <= 0:
        raise ValueError("beta must be positive")

    def f(cost):
        c = _as_float(cost)
        with np.errstate(over="ignore"):
            w = np.exp(-beta * c)
        return np.where(c <= threshold, w, 0.0)

    return f
