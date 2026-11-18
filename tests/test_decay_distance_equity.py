import networkx as nx
import numpy as np
import pandas as pd
import pytest

from healthaccess import decay, distance, equity


def test_binary_decay():
    f = decay.binary(10)
    assert f([0, 10, 10.01, np.inf, np.nan]).tolist() == [1, 1, 0, 0, 0]


def test_stepwise_decay_zone_edges_are_inclusive():
    f = decay.stepwise([10, 20, 30], [1.0, 0.68, 0.22])
    assert f([0, 10, 10.5, 20, 30, 30.1]).tolist() == [1.0, 1.0, 0.68, 0.68, 0.22, 0.0]


def test_stepwise_validates():
    with pytest.raises(ValueError):
        decay.stepwise([20, 10], [1, 0.5])
    with pytest.raises(ValueError):
        decay.stepwise([10, 20], [1])


def test_gaussian_decay_endpoints():
    f = decay.gaussian(30)
    w = f([0, 15, 30, 31])
    assert w[0] == pytest.approx(1.0)
    assert 0 < w[1] < 1
    assert w[2] == pytest.approx(0.0, abs=1e-12)
    assert w[3] == 0.0


def test_power_decay_clamps_small_costs():
    f = decay.power(2.0, min_cost=1.0)
    assert f([0.0, 0.5, 2.0]).tolist() == [1.0, 1.0, 0.25]


def test_haversine_one_degree_latitude():
    d = distance.haversine_matrix([0.0], [0.0], [1.0], [0.0])
    assert d.iloc[0, 0] == pytest.approx(111.195, abs=0.01)


def test_euclidean_matrix():
    d = distance.euclidean_matrix([[0, 0]], [[3, 4], [0, 1]], ["a"], ["x", "y"])
    assert d.loc["a", "x"] == 5.0 and d.loc["a", "y"] == 1.0


def test_from_long_keeps_minimum():
    t = pd.DataFrame({"origin": ["a", "a", "b"], "destination": ["x", "x", "x"], "cost": [5, 3, 7]})
    m = distance.from_long(t)
    assert m.loc["a", "x"] == 3 and m.loc["b", "x"] == 7


def test_network_matrix_respects_one_way_streets():
    g = nx.DiGraph()
    g.add_edge(1, 2, travel_time=5)  # 1 -> 2 is one-way
    g.add_edge(2, 3, travel_time=5)
    g.add_edge(3, 1, travel_time=20)
    m = distance.network_cost_matrix(g, {"home": 1}, {"clinic": 3})
    assert m.loc["home", "clinic"] == 10  # 1 -> 2 -> 3
    back = distance.network_cost_matrix(g, {"home": 3}, {"clinic": 1})
    assert back.loc["home", "clinic"] == 20  # must take the 3 -> 1 edge


def test_network_cutoff_gives_nan():
    g = nx.path_graph(5)
    nx.set_edge_attributes(g, 10, "travel_time")
    m = distance.network_cost_matrix(g, {"far": 4, "near": 1}, {"c": 0}, cutoff=15)
    assert m.loc["near", "c"] == 10 and np.isnan(m.loc["far", "c"])


def test_gini():
    assert equity.weighted_gini([5, 5, 5]) == pytest.approx(0.0)
    assert equity.weighted_gini([0, 0, 0, 1]) == pytest.approx(0.75)
    # weights act like repeated observations
    assert equity.weighted_gini([0, 1], [3, 1]) == pytest.approx(equity.weighted_gini([0, 0, 0, 1]))


def test_share_below_and_groups():
    access = pd.Series({"a": 0.1, "b": 0.5, "c": 0.9})
    pop = pd.Series({"a": 100, "b": 100, "c": 200})
    assert equity.share_below(access, pop, 0.6) == pytest.approx(0.5)
    groups = pd.Series({"a": "rural", "b": "rural", "c": "urban"})
    s = equity.group_summary(access, pop, groups)
    assert s.loc["rural", "weighted_mean_access"] == pytest.approx(0.3)
    assert s.loc["urban", "population"] == 200


def test_weighted_gini_matches_mean_absolute_difference_formula():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.gamma(2.0, 1.0, 60)
        w = rng.integers(1, 1000, 60).astype(float)
        total, mean = w.sum(), (w * x).sum() / w.sum()
        g = (w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :])).sum() / (2 * total**2 * mean)
        assert equity.weighted_gini(x, w) == pytest.approx(g, rel=1e-12)


def test_missing_access_is_left_out_of_equity_summaries():
    access = pd.Series({"a": 0.1, "b": np.nan, "c": 0.9})
    pop = pd.Series({"a": 100, "b": 100, "c": 200})
    groups = pd.Series({"a": "rural", "b": "rural", "c": "urban"})
    assert equity.share_below(access, pop, 0.5) == pytest.approx(100 / 300)
    s = equity.group_summary(access, pop, groups)
    assert s.loc["rural", "weighted_mean_access"] == pytest.approx(0.1)
    assert s.loc["rural", "population"] == 100


def test_haversine_exact_geometry():
    # equator to pole is a quarter of a great circle: pi * R / 2
    d = distance.haversine_matrix([0.0], [0.0], [90.0], [0.0])
    assert d.iloc[0, 0] == pytest.approx(np.pi * distance.EARTH_RADIUS_KM / 2, rel=1e-12)
    # antipodes are half a great circle
    d = distance.haversine_matrix([0.0], [0.0], [0.0], [180.0])
    assert d.iloc[0, 0] == pytest.approx(np.pi * distance.EARTH_RADIUS_KM, rel=1e-12)
    # symmetric and zero on the diagonal
    m = distance.haversine_matrix([33.6, 34.0], [-84.4, -83.3], [33.6, 34.0], [-84.4, -83.3])
    assert np.allclose(np.diag(m.to_numpy()), 0.0) and np.allclose(m.to_numpy(), m.to_numpy().T)
