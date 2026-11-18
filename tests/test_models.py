import numpy as np
import pandas as pd
import pytest

import healthaccess as ha
from healthaccess import decay


@pytest.fixture
def toy():
    """Two neighborhoods, two clinics. Worked by hand in the README."""
    demand = pd.Series({"A": 100.0, "B": 300.0})
    supply = pd.Series({"X": 2.0, "Y": 1.0})
    cost = pd.DataFrame({"X": [5.0, 15.0], "Y": [25.0, 8.0]}, index=["A", "B"])
    return demand, supply, cost


def test_two_step_fca_matches_hand_calculation(toy):
    demand, supply, cost = toy
    access = ha.two_step_fca(demand, supply, cost, threshold=20)
    # Clinic X serves A and B (400 people): 2 / 400 = 0.005
    # Clinic Y serves only B (300 people):  1 / 300
    assert access["A"] == pytest.approx(0.005)
    assert access["B"] == pytest.approx(0.005 + 1 / 300)


def test_provider_ratios(toy):
    demand, supply, cost = toy
    r = ha.provider_to_population_ratios(demand, supply, cost, decay.binary(20))
    assert r["X"] == pytest.approx(2 / 400)
    assert r["Y"] == pytest.approx(1 / 300)


@pytest.mark.parametrize(
    "model",
    [
        lambda d, s, c: ha.two_step_fca(d, s, c, threshold=30),
        lambda d, s, c: ha.enhanced_2sfca(d, s, c),
        lambda d, s, c: ha.gaussian_2sfca(d, s, c, threshold=30),
        lambda d, s, c: ha.gravity(d, s, c, beta=1.5),
    ],
)
def test_supply_is_conserved(model):
    rng = np.random.default_rng(42)
    n_d, n_s = 60, 8
    demand = pd.Series(rng.integers(100, 5000, n_d).astype(float), index=[f"t{i}" for i in range(n_d)])
    supply = pd.Series(rng.integers(1, 20, n_s).astype(float), index=[f"c{j}" for j in range(n_s)])
    # every clinic has at least one tract within 5 minutes so all supply is used
    cost = pd.DataFrame(rng.uniform(0, 60, (n_d, n_s)), index=demand.index, columns=supply.index)
    cost.iloc[0, :] = 1.0
    access = model(demand, supply, cost)
    assert (access * demand).sum() == pytest.approx(supply.sum())


def test_isolated_neighborhood_gets_zero(toy):
    demand, supply, cost = toy
    demand = pd.concat([demand, pd.Series({"C": 50.0})])
    access = ha.two_step_fca(demand, supply, cost, threshold=20)  # C missing from cost
    assert access["C"] == 0.0
    assert access["A"] == pytest.approx(0.005)


def test_nan_cost_is_unreachable(toy):
    demand, supply, cost = toy
    cost.loc["B", "Y"] = np.nan
    access = ha.two_step_fca(demand, supply, cost, threshold=20)
    assert access["B"] == pytest.approx(0.005)


def test_clinic_without_demand_is_ignored(toy):
    demand, supply, cost = toy
    supply = pd.concat([supply, pd.Series({"Z": 10.0})])
    cost["Z"] = 999.0
    access = ha.two_step_fca(demand, supply, cost, threshold=20)
    assert access["A"] == pytest.approx(0.005)


def test_gravity_equals_fca_with_power_decay(toy):
    demand, supply, cost = toy
    g = ha.gravity(demand, supply, cost, beta=2.0)
    f = ha.fca(demand, supply, cost, decay.power(2.0))
    pd.testing.assert_series_equal(g, f, check_names=False)


def test_nearest_and_within(toy):
    demand, supply, cost = toy
    near = ha.nearest_provider_cost(demand, supply, cost)
    assert near.tolist() == [5.0, 8.0]
    within = ha.providers_within(demand, supply, cost, threshold=10)
    assert within.tolist() == [2.0, 1.0]


def test_rejects_bad_inputs(toy):
    demand, supply, cost = toy
    with pytest.raises(ValueError):
        ha.two_step_fca(-demand, supply, cost, threshold=20)
    with pytest.raises(TypeError):
        ha.two_step_fca(demand.to_numpy(), supply, cost, threshold=20)
    with pytest.raises(ValueError):
        ha.two_step_fca(demand, supply, -cost, threshold=20)


def test_three_step_fca_matches_hand_calculation(toy):
    demand, supply, cost = toy
    access = ha.three_step_fca(demand, supply, cost, decay_fn=decay.binary(20))
    # A reaches only X (G=1); B reaches X and Y (G=0.5 each).
    # R_X = 2 / (1*100 + 0.5*300) = 0.008 ; R_Y = 1 / (0.5*300) = 1/150
    assert access["A"] == pytest.approx(0.008)
    assert access["B"] == pytest.approx(0.5 * 0.008 + 0.5 / 150)
    # supply conserved: 100*0.008 + 300*(0.004 + 1/300) = 3 physicians
    assert (access * demand).sum() == pytest.approx(3.0)


def test_three_step_fca_default_is_stepwise_and_conserves_supply(toy):
    demand, supply, cost = toy
    access = ha.three_step_fca(demand, supply, cost)
    assert (access * demand).sum() == pytest.approx(supply.sum())


def test_modified_2sfca_squares_second_step_weights(toy):
    demand, supply, cost = toy
    f = decay.stepwise([10, 20, 30], [1.0, 0.5, 0.25])
    access = ha.modified_2sfca(demand, supply, cost, decay_fn=f)
    # W: A-X 1, A-Y 0.25, B-X 0.5, B-Y 1
    r_x, r_y = 2 / (100 * 1 + 300 * 0.5), 1 / (100 * 0.25 + 300 * 1)
    assert access["A"] == pytest.approx(r_x * 1 + r_y * 0.25**2)
    assert access["B"] == pytest.approx(r_x * 0.5**2 + r_y * 1)
    # with binary decay it equals classic 2SFCA
    m = ha.modified_2sfca(demand, supply, cost, decay_fn=decay.binary(20))
    assert m.to_numpy() == pytest.approx(ha.two_step_fca(demand, supply, cost, 20).to_numpy())


def test_modified_2sfca_needs_threshold_or_decay(toy):
    with pytest.raises(ValueError):
        ha.modified_2sfca(*toy)
