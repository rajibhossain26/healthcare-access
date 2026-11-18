"""Summaries for how evenly accessibility is shared across a population."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_gini(values, weights=None) -> float:
    """Population-weighted Gini coefficient (0 = perfectly equal, 1 = maximal
    inequality)."""
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    mask = ~np.isnan(x) & (w > 0)
    x, w = x[mask], w[mask]
    if x.size == 0 or np.sum(w * x) == 0:
        return 0.0
    if np.any(x < 0):
        raise ValueError("Gini is defined for non-negative values only")
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_w = np.cumsum(w)
    cum_wx = np.cumsum(w * x)
    total_w, total_wx = cum_w[-1], cum_wx[-1]
    # Area under the Lorenz curve by the trapezoid rule.
    prev_wx = np.concatenate([[0.0], cum_wx[:-1]])
    lorenz_area = np.sum(w / total_w * (prev_wx + cum_wx) / (2 * total_wx))
    return float(1.0 - 2.0 * lorenz_area)


def share_below(access: pd.Series, population: pd.Series, threshold: float) -> float:
    """Share of the population living where accessibility is below a threshold
    (e.g. a federal shortage-area ratio)."""
    pop = population.reindex(access.index).astype(float)
    total = pop.sum()
    if total == 0:
        return 0.0
    return float(pop[access < threshold].sum() / total)


def group_summary(access: pd.Series, population: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Population-weighted mean accessibility and population for each group
    (e.g. urban vs rural, or income quintile)."""
    df = pd.DataFrame({"access": access, "pop": population.reindex(access.index), "group": groups.reindex(access.index)})
    df = df.dropna(subset=["group"])
    df["wx"] = df["access"] * df["pop"]
    out = df.groupby("group", observed=True).agg(population=("pop", "sum"), wx=("wx", "sum"))
    out["weighted_mean_access"] = out["wx"] / out["population"]
    return out.drop(columns="wx")
