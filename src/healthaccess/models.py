"""Spatial accessibility models.

All models take

* ``demand``: a pandas Series of population indexed by demand-location id
  (e.g. census tract GEOID),
* ``supply``: a pandas Series of provider capacity indexed by provider id
  (e.g. number of physicians or beds),
* ``cost``: a DataFrame of travel costs with demand ids as the index and
  provider ids as the columns. Missing pairs (NaN) are treated as unreachable.

and return a pandas Series of accessibility scores indexed like ``demand``.
Scores from the FCA family are "providers per person"; multiply by e.g. 1,000
or 100,000 for a readable ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import decay as _decay

__all__ = [
    "align",
    "provider_to_population_ratios",
    "fca",
    "two_step_fca",
    "enhanced_2sfca",
    "gaussian_2sfca",
    "gravity",
    "nearest_provider_cost",
    "providers_within",
]


def align(demand: pd.Series, supply: pd.Series, cost: pd.DataFrame):
    """Validate inputs and return aligned numpy arrays.

    Returns ``(population, capacity, cost_matrix)`` where the cost matrix has
    shape (n_demand, n_supply) in the order of ``demand.index`` and
    ``supply.index``. Ids missing from ``cost`` become unreachable (inf).
    """
    if not isinstance(demand, pd.Series) or not isinstance(supply, pd.Series):
        raise TypeError("demand and supply must be pandas Series indexed by id")
    if not isinstance(cost, pd.DataFrame):
        raise TypeError("cost must be a DataFrame (demand ids x supply ids)")
    if demand.index.has_duplicates or supply.index.has_duplicates:
        raise ValueError("demand and supply ids must be unique")
    pop = demand.astype(float).to_numpy()
    cap = supply.astype(float).to_numpy()
    if np.any(pop < 0) or np.any(cap < 0):
        raise ValueError("population and capacity must be non-negative")
    if np.isnan(pop).any() or np.isnan(cap).any():
        raise ValueError("population and capacity must not contain NaN")

    matrix = cost.reindex(index=demand.index, columns=supply.index).to_numpy(dtype=float)
    matrix = np.where(np.isnan(matrix), np.inf, matrix)
    if np.any(matrix < 0):
        raise ValueError("travel costs must be non-negative")
    return pop, cap, matrix


def provider_to_population_ratios(
    demand: pd.Series, supply: pd.Series, cost: pd.DataFrame, decay_fn
) -> pd.Series:
    """Step 1 of an FCA model: each provider's capacity divided by the
    distance-weighted population inside its catchment."""
    pop, cap, matrix = align(demand, supply, cost)
    w = decay_fn(matrix)
    weighted_demand = (pop[:, None] * w).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(weighted_demand > 0, cap / weighted_demand, 0.0)
    return pd.Series(ratio, index=supply.index, name="provider_ratio")


def fca(
    demand: pd.Series,
    supply: pd.Series,
    cost: pd.DataFrame,
    decay_fn,
    name: str = "access",
) -> pd.Series:
    """Generic two-step floating catchment area model with any decay function.

    Step 1: R_j = S_j / sum_i P_i w(d_ij)
    Step 2: A_i = sum_j R_j w(d_ij)

    Using the same weights in both steps keeps total supply conserved:
    sum_i P_i A_i equals the capacity of every provider that has demand.
    """
    pop, cap, matrix = align(demand, supply, cost)
    w = decay_fn(matrix)
    weighted_demand = (pop[:, None] * w).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(weighted_demand > 0, cap / weighted_demand, 0.0)
    access = w @ ratio
    return pd.Series(access, index=demand.index, name=name)


def two_step_fca(demand, supply, cost, threshold: float) -> pd.Series:
    """Classic 2SFCA (Luo & Wang 2003): equal weight inside the catchment."""
    return fca(demand, supply, cost, _decay.binary(threshold), name="access_2sfca")


def enhanced_2sfca(
    demand,
    supply,
    cost,
    breaks=(10.0, 20.0, 30.0),
    weights=(1.0, 0.68, 0.22),
) -> pd.Series:
    """Enhanced 2SFCA (Luo & Qi 2009) with stepwise travel-time zones.

    The default zones are 0-10, 10-20 and 20-30 minutes with the Gaussian-
    derived weights used in the original paper.
    """
    return fca(demand, supply, cost, _decay.stepwise(breaks, weights), name="access_e2sfca")


def gaussian_2sfca(demand, supply, cost, threshold: float) -> pd.Series:
    """Kernel density 2SFCA with Gaussian decay (Dai 2010)."""
    return fca(demand, supply, cost, _decay.gaussian(threshold), name="access_g2sfca")


def gravity(demand, supply, cost, beta: float = 1.5, min_cost: float = 1.0) -> pd.Series:
    """Gravity-based accessibility with competition (Joseph & Bantock 1982).

    A_i = sum_j S_j d_ij^-beta / V_j, with V_j = sum_k P_k d_kj^-beta.
    Equivalent to an FCA model with power decay and no catchment limit.
    """
    return fca(demand, supply, cost, _decay.power(beta, min_cost=min_cost), name="access_gravity")


def nearest_provider_cost(demand, supply, cost) -> pd.Series:
    """Travel cost from each demand location to its nearest provider
    (providers with zero capacity are ignored)."""
    _, cap, matrix = align(demand, supply, cost)
    usable = np.where(cap[None, :] > 0, matrix, np.inf)
    nearest = usable.min(axis=1) if usable.shape[1] else np.full(len(demand), np.inf)
    return pd.Series(nearest, index=demand.index, name="nearest_cost")


def providers_within(demand, supply, cost, threshold: float) -> pd.Series:
    """Total provider capacity reachable within ``threshold`` (a cumulative
    opportunities measure, ignoring competition)."""
    _, cap, matrix = align(demand, supply, cost)
    reachable = matrix <= threshold
    return pd.Series(reachable @ cap, index=demand.index, name="capacity_within")
