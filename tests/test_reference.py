"""Compare every model with an independent loop-based implementation of the
published equations on many random problems, including awkward ones
(unreachable pairs, empty neighbourhoods, providers nobody can reach)."""

import math

import numpy as np
import pandas as pd
import pytest

import healthaccess as ha
from healthaccess import decay

import reference as ref


def random_problem(seed, n=25, m=8):
    rng = np.random.default_rng(seed)
    P = rng.integers(0, 5000, n).astype(float)
    P[rng.random(n) < 0.1] = 0.0                       # some empty neighbourhoods
    S = rng.integers(0, 12, m).astype(float)
    cost = rng.uniform(0.0, 45.0, (n, m))
    cost[rng.random((n, m)) < 0.15] = np.nan           # some unreachable pairs
    cost[:, -1] = np.nan                               # a provider nobody can reach
    demand = pd.Series(P, index=[f"t{i}" for i in range(n)])
    supply = pd.Series(S, index=[f"c{j}" for j in range(m)])
    matrix = pd.DataFrame(cost, index=demand.index, columns=supply.index)
    return demand, supply, matrix


CASES = [
    ("2SFCA", lambda d, s, c: ha.two_step_fca(d, s, c, threshold=30), ref.ref_2sfca, ref.binary(30)),
    ("E2SFCA", lambda d, s, c: ha.enhanced_2sfca(d, s, c), ref.ref_2sfca, ref.e2sfca_zones),
    ("G2SFCA", lambda d, s, c: ha.gaussian_2sfca(d, s, c, threshold=30), ref.ref_2sfca, ref.dai_gaussian(30)),
    ("gravity", lambda d, s, c: ha.gravity(d, s, c, beta=1.5), ref.ref_2sfca, ref.gravity(1.5)),
    ("3SFCA", lambda d, s, c: ha.three_step_fca(d, s, c), ref.ref_3sfca, ref.e2sfca_zones),
    ("3SFCA-gauss", lambda d, s, c: ha.three_step_fca(d, s, c, decay_fn=decay.gaussian(30)),
     ref.ref_3sfca, ref.dai_gaussian(30)),
    ("M2SFCA", lambda d, s, c: ha.modified_2sfca(d, s, c, threshold=30), ref.ref_m2sfca, ref.dai_gaussian(30)),
]


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("name,model,reference_model,f", CASES, ids=[c[0] for c in CASES])
def test_model_matches_reference(name, model, reference_model, f, seed):
    demand, supply, cost = random_problem(seed)
    got = model(demand, supply, cost).to_numpy()
    W = ref.weights(cost.to_numpy().tolist(), f)
    want = reference_model(demand.tolist(), supply.tolist(), W)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("seed", range(10))
def test_order_of_ids_does_not_matter(seed):
    demand, supply, cost = random_problem(seed)
    a = ha.enhanced_2sfca(demand, supply, cost)
    shuffled = cost.sample(frac=1, random_state=seed).sample(frac=1, axis=1, random_state=seed)
    b = ha.enhanced_2sfca(demand.sample(frac=1, random_state=1), supply.sample(frac=1, random_state=2), shuffled)
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


@pytest.mark.parametrize("seed", range(10))
def test_supply_conserved_on_random_problems(seed):
    demand, supply, cost = random_problem(seed)
    reachable_with_demand = [
        s for j, s in enumerate(supply)
        if (demand.to_numpy() * (cost.iloc[:, j].to_numpy() <= 30)).sum() > 0
    ]
    for model in (lambda d, s, c: ha.two_step_fca(d, s, c, 30),
                  lambda d, s, c: ha.three_step_fca(d, s, c, decay_fn=decay.binary(30))):
        total = float((model(demand, supply, cost) * demand).sum())
        assert total == pytest.approx(sum(reachable_with_demand), rel=1e-12)


def test_doubling_everything_keeps_ratios():
    demand, supply, cost = random_problem(3)
    a = ha.two_step_fca(demand, supply, cost, 30)
    b = ha.two_step_fca(demand * 2, supply * 2, cost, 30)
    assert a.to_numpy() == pytest.approx(b.to_numpy())


def test_gravity_scale_of_decay_cancels():
    # Multiplying every weight by a constant must not change FCA-family scores.
    demand, supply, cost = random_problem(4)
    f = decay.power(1.5)
    a = ha.fca(demand, supply, cost, f)
    b = ha.fca(demand, supply, cost, lambda c: 7.0 * f(c))
    assert a.to_numpy() == pytest.approx(b.to_numpy())


def test_e2sfca_default_weights_are_luo_qi_2009():
    f = decay.stepwise([10, 20, 30], [1.0, 0.68, 0.22])
    for d in [0, 3.3, 10, 10.0001, 19.9, 20, 25, 30, 30.0001, 50]:
        assert f([d])[0] == pytest.approx(ref.e2sfca_zones(d))


def test_dai_gaussian_matches_formula():
    f = decay.gaussian(30)
    for d in [0, 1, 7.5, 15, 29.99, 30, 31]:
        assert f([d])[0] == pytest.approx(ref.dai_gaussian(30)(d), abs=1e-14)
    assert math.isclose(f([0])[0], 1.0)
