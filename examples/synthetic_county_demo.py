"""Compare accessibility models on a synthetic county.

The county is made up: a dense urban core with most clinics, and a sparse
rural periphery. It shows how the choice of model changes the picture,
without needing any downloads. Run from the repository root:

    python examples/synthetic_county_demo.py

To use real data, replace ``demand``/``supply`` with census tract populations
and provider counts, and ``cost`` with a drive-time matrix, for example from
``healthaccess.distance.osm_drive_cost_matrix``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import healthaccess as ha

rng = np.random.default_rng(7)

# 30 x 30 grid of 2 km cells (a 60 km wide county)
n = 30
xs, ys = np.meshgrid(np.arange(n) * 2.0 + 1, np.arange(n) * 2.0 + 1)
cells = pd.DataFrame({"x": xs.ravel(), "y": ys.ravel()})
cells.index = [f"tract_{i:03d}" for i in range(len(cells))]

centre = np.array([30.0, 30.0])
r = np.hypot(cells.x - centre[0], cells.y - centre[1])
density = 4000 * np.exp(-r / 8) + 150  # people per cell
demand = pd.Series(rng.poisson(density).astype(float), index=cells.index, name="population")

# 14 clinics: most in the core, a few in small towns
core = centre + rng.normal(0, 5, (10, 2))
towns = np.array([[8.0, 10.0], [52.0, 12.0], [50.0, 50.0], [10.0, 48.0]])
clinic_xy = np.vstack([core, towns])
supply = pd.Series(
    np.concatenate([rng.integers(30, 90, 10), rng.integers(3, 10, 4)]).astype(float),
    index=[f"clinic_{j:02d}" for j in range(len(clinic_xy))],
    name="physicians",
)

# Drive time in minutes: straight-line km x 1.3 detour factor at 50 km/h
km = ha.distance.euclidean_matrix(cells[["x", "y"]].to_numpy(), clinic_xy, cells.index, supply.index)
minutes = km * 1.3 / 50 * 60

per_1000 = 1000
results = pd.DataFrame(
    {
        "2SFCA (30 min)": ha.two_step_fca(demand, supply, minutes, threshold=30) * per_1000,
        "Enhanced 2SFCA": ha.enhanced_2sfca(demand, supply, minutes) * per_1000,
        "Gaussian 2SFCA (30 min)": ha.gaussian_2sfca(demand, supply, minutes, threshold=30) * per_1000,
        "Gravity (beta = 1.5)": ha.gravity(demand, supply, minutes, beta=1.5) * per_1000,
    }
)

# Federal primary-care shortage areas use roughly 1 physician per 3,500 people
shortage = 1000 / 3500

print(f"Total population {demand.sum():,.0f}, physicians {supply.sum():.0f}")
print("Physicians per 1,000 residents, population-weighted summary\n")
for name, col in results.items():
    gini = ha.equity.weighted_gini(col, demand)
    low = ha.equity.share_below(col, demand, shortage)
    print(f"{name:26s} Gini = {gini:.3f}   in shortage (<1:3,500) = {low:6.1%}")

fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), constrained_layout=True)
grid = lambda s: s.to_numpy().reshape(n, n)
extent = (0, 2 * n, 0, 2 * n)

im = axes[0].imshow(grid(demand), origin="lower", extent=extent, cmap="Greys")
axes[0].set_title("Population (synthetic)")
fig.colorbar(im, ax=axes[0], shrink=0.8)

vmax = np.nanpercentile(results.to_numpy(), 99)
for ax, (name, col) in zip(axes[1:], results.items()):
    im = ax.imshow(grid(col), origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(name)
fig.colorbar(im, ax=axes[1:], shrink=0.8, label="Physicians per 1,000 residents")

for ax in axes:
    ax.scatter(clinic_xy[:, 0], clinic_xy[:, 1], s=supply * 3, c="#e4572e", edgecolor="white", linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])

out = Path(__file__).resolve().parents[1] / "docs" / "model_comparison.png"
fig.savefig(out, dpi=90)
print(f"\nSaved {out.relative_to(out.parents[1])}")
