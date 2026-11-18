"""Cross-check healthcare-access against two independent packages.

* PySAL ``access`` (Saxon et al.): 2SFCA, Enhanced 2SFCA, 3SFCA, gravity.
* Aceso (Lewis): 2SFCA, gravity and Modified 2SFCA (M2SFCA).

To compare the *models* rather than incidental choices, both sides are given
exactly the same distance-decay function (the other package's own function
is wrapped and passed to healthcare-access). A second set of checks confirms
that the built-in decay functions of healthcare-access give the same weights
as the other packages' named functions.

Aceso's 3SFCA is not compared: it computes the selection (Huff) probabilities
from inverse distance rather than from the decay weights, a known shortcut
marked FIXME in its source, so it is a different model from Wan et al. (2012).

Run:  python validation/compare_external.py   (needs: pip install access aceso)
Writes validation/RESULTS.md and exits non-zero if any check fails.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import healthaccess as ha
from healthaccess import decay

TOL = 1e-9  # relative tolerance
ROOT = Path(__file__).resolve().parent
rows: list[dict] = []


def problem(seed: int, n: int = 120, m: int = 25):
    rng = np.random.default_rng(seed)
    P = rng.integers(50, 8000, n).astype(float)
    S = rng.integers(1, 15, m).astype(float)
    cost = rng.uniform(0.5, 45.0, (n, m))           # continuous: no ties at zone edges
    cost[rng.random((n, m)) < 0.2] = np.nan         # unreachable pairs
    d = pd.Series(P, index=pd.Index([f"t{i:03d}" for i in range(n)], name="geoid"))
    s = pd.Series(S, index=pd.Index([f"c{j:02d}" for j in range(m)], name="geoid"))
    c = pd.DataFrame(cost, index=d.index, columns=s.index)
    return d, s, c


def wrap(fn, max_cost=None):
    """Turn another package's scalar/array decay into a healthcare-access decay
    (0 for unreachable pairs and beyond max_cost)."""
    vec = np.vectorize(lambda x: float(fn(x)), otypes=[float])

    def f(cost):
        c = np.asarray(cost, dtype=float)
        ok = np.isfinite(c) & (True if max_cost is None else (c <= max_cost))
        out = np.zeros_like(c)
        if ok.any():
            out[ok] = vec(c[ok])
        return out

    return f


def compare(label, ours: pd.Series, theirs: pd.Series, note=""):
    theirs = theirs.reindex(ours.index).astype(float).fillna(0.0)
    a, b = ours.to_numpy(float), theirs.to_numpy(float)
    scale = max(np.abs(b).max(), 1e-300)
    max_rel = float(np.abs(a - b).max() / scale)
    ok = bool(np.allclose(a, b, rtol=TOL, atol=TOL * scale))
    rows.append({"check": label, "locations": len(a), "max_rel_diff": max_rel, "pass": ok, "note": note})
    print(f"{'PASS' if ok else 'FAIL'}  {label:58s} max rel diff {max_rel:.2e}  {note}")


# ---------------------------------------------------------------- PySAL access
def pysal_checks(seeds=(1, 2, 3)):
    import access
    from access import Access, weights

    print("access version:", getattr(access, "__version__", "?"))
    for name in ("two_stage_fca", "three_stage_fca", "weighted_catchment"):
        fn = getattr(access.fca, name, None)
        if fn is not None:
            print(f"\n----- source of access.fca.{name} -----\n{inspect.getsource(fn)}")
    for name in ("step_fn", "gravity", "gaussian"):
        print(f"\n----- source of access.weights.{name} -----\n{inspect.getsource(getattr(weights, name))}")

    for seed in seeds:
        d, s, c = problem(seed)
        long = (c.rename_axis(index="origin", columns="dest").stack().rename("cost")
                .reset_index().dropna(subset=["cost"]))
        A = Access(demand_df=d.to_frame("pop"), demand_index=True, demand_value="pop",
                   supply_df=s.to_frame("doc"), supply_index=True, supply_value="doc",
                   cost_df=long, cost_origin="origin", cost_dest="dest", cost_name="cost")

        def col(df):
            return df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df

        # 2SFCA, 30-minute catchment
        theirs = col(A.two_stage_fca(name="t2", max_cost=30))
        compare(f"PySAL 2SFCA (30 min), seed {seed}", ha.two_step_fca(d, s, c, threshold=30), theirs)

        # Enhanced 2SFCA with the Luo & Qi (2009) zones, their step function on both sides
        step = weights.step_fn({10: 1, 20: 0.68, 30: 0.22})
        theirs = col(A.enhanced_two_stage_fca(name="e2", weight_fn=step, max_cost=30))
        compare(f"PySAL E2SFCA, seed {seed}", ha.fca(d, s, c, wrap(step, 30)), theirs)
        compare(f"PySAL E2SFCA vs built-in enhanced_2sfca, seed {seed}", ha.enhanced_2sfca(d, s, c), theirs,
                "built-in zone weights")

        # 3SFCA with the same step weights
        theirs = col(A.three_stage_fca(name="t3", weight_fn=step, max_cost=30))
        compare(f"PySAL 3SFCA, seed {seed}", ha.three_step_fca(d, s, c, decay_fn=wrap(step, 30)), theirs)

        # Gravity (no catchment limit), PySAL's gravity weights on both sides
        grav = weights.gravity(scale=1, alpha=-1.5, min_dist=1)
        theirs = col(A.two_stage_fca(name="g", weight_fn=grav, max_cost=1e9))
        compare(f"PySAL gravity (beta 1.5), seed {seed}", ha.fca(d, s, c, wrap(grav)), theirs)
        compare(f"PySAL gravity vs built-in gravity, seed {seed}", ha.gravity(d, s, c, beta=1.5, min_cost=1), theirs,
                "built-in power decay")

        # Gaussian weights, PySAL's own Gaussian function on both sides
        gauss = weights.gaussian(sigma=15)
        theirs = col(A.two_stage_fca(name="ga", weight_fn=gauss, max_cost=30))
        compare(f"PySAL 2SFCA with Gaussian weights, seed {seed}", ha.fca(d, s, c, wrap(gauss, 30)), theirs)

    # named decay functions give the same weights
    x = np.linspace(0.01, 45, 2000)
    for label, mine, theirs_fn in [
        ("decay.stepwise == access step_fn (Luo & Qi zones)", decay.stepwise([10, 20, 30], [1, .68, .22]),
         wrap(weights.step_fn({10: 1, 20: 0.68, 30: 0.22}), 30)),
        ("decay.power == access gravity (beta 1.5, min 1)", decay.power(1.5, 1.0), wrap(weights.gravity(1, -1.5, 1))),
    ]:
        compare(label, pd.Series(mine(x)), pd.Series(theirs_fn(x)), "weights on a 2,000-point grid")


# ---------------------------------------------------------------------- Aceso
def aceso_checks(seeds=(1, 2, 3)):
    import aceso

    for seed in seeds:
        d, s, c = problem(seed)
        dist = c.to_numpy(float)
        dist = np.where(np.isnan(dist), np.inf, dist)
        P, S = d.to_numpy(float), s.to_numpy(float)

        g30 = decay.gaussian(30)
        ref = lambda arr: g30(arr)  # noqa: E731  same weights on both sides

        m = aceso.GravityModel(decay_function=ref, suboptimality_exponent=2.0)
        theirs = pd.Series(m.calculate_accessibility_scores(distance_matrix=dist, demand_array=P, supply_array=S),
                           index=d.index)
        compare(f"Aceso M2SFCA (Gaussian 30), seed {seed}", ha.modified_2sfca(d, s, c, threshold=30), theirs)

        m = aceso.TwoStepFCA(radius=30.0)
        theirs = pd.Series(m.calculate_accessibility_scores(distance_matrix=dist, demand_array=P, supply_array=S),
                           index=d.index)
        compare(f"Aceso 2SFCA (30 min), seed {seed}", ha.two_step_fca(d, s, c, threshold=30), theirs)

        pw = decay.power(1.5, 1.0)
        m = aceso.GravityModel(decay_function=lambda arr: pw(arr))
        theirs = pd.Series(m.calculate_accessibility_scores(distance_matrix=dist, demand_array=P, supply_array=S),
                           index=d.index)
        compare(f"Aceso gravity (beta 1.5), seed {seed}", ha.gravity(d, s, c, beta=1.5), theirs)


def write_report(versions):
    lines = [
        "# External validation",
        "",
        "Generated by [`compare_external.py`](compare_external.py) in the `validate` GitHub Actions workflow.",
        f"Tolerance: relative difference below {TOL:g}. Each problem has 120 demand locations, 25 providers and "
        "20% unreachable pairs.",
        "",
        "Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()),
        "",
        "| Check | Locations | Max relative difference | Result |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['check']}{' (' + r['note'] + ')' if r['note'] else ''} | {r['locations']} | "
                     f"{r['max_rel_diff']:.1e} | {'pass' if r['pass'] else '**FAIL**'} |")
    lines += [
        "",
        f"**{sum(r['pass'] for r in rows)} of {len(rows)} checks passed.**",
        "",
        "Not compared: Aceso's 3SFCA, which derives its selection weights from inverse distance rather than "
        "from the decay function (see the module docstring).",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    from importlib.metadata import version

    pysal_checks()
    aceso_checks()
    versions = {p: version(p) for p in ("healthcare-access", "access", "aceso", "numpy", "pandas")}
    write_report(versions)
    failed = [r for r in rows if not r["pass"]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} checks passed")
    sys.exit(1 if failed else 0)
