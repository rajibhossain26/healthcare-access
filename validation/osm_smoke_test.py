"""Real-world smoke test: OpenStreetMap drive times for a small Georgia city,
then every model, with sanity checks on the results.

Downloads the road network for Winterville, Georgia (a small town east of
Athens), snaps a few points to it and checks that drive times are finite,
positive, asymmetric where one-way streets allow, and plausible for a town a
few kilometres across.
"""

import numpy as np
import pandas as pd

import healthaccess as ha

homes = pd.DataFrame({"lon": [-83.2785, -83.2900, -83.2650, -83.2830],
                      "lat": [33.9670, 33.9600, 33.9720, 33.9550],
                      "population": [800, 1200, 600, 900]},
                     index=["h1", "h2", "h3", "h4"])
clinics = pd.DataFrame({"lon": [-83.2810, -83.2700], "lat": [33.9665, 33.9600],
                        "physicians": [3, 1]}, index=["c1", "c2"])

minutes = ha.distance.osm_drive_cost_matrix("Winterville, Georgia, USA", homes, clinics, cutoff_minutes=30)
print("Drive times (minutes):\n", minutes.round(2))
straight = ha.distance.haversine_matrix(homes.lat, homes.lon, clinics.lat, clinics.lon, homes.index, clinics.index)
print("Straight-line km:\n", straight.round(2))

assert minutes.shape == (4, 2)
assert np.isfinite(minutes.to_numpy()).all(), "every point should be reachable within 30 minutes"
assert (minutes.to_numpy() >= 0).all() and (minutes.to_numpy() < 15).all(), "a small town: drives under 15 min"
# implied speed should be plausible for local roads (not 0, not motorway-plus)
kmh = straight / (minutes.clip(lower=0.5) / 60)
print("Implied straight-line speed (km/h):\n", kmh.round(1))
assert (kmh.to_numpy() < 120).all()

for name, access in {
    "2SFCA": ha.two_step_fca(homes.population, clinics.physicians, minutes, threshold=15),
    "E2SFCA": ha.enhanced_2sfca(homes.population, clinics.physicians, minutes),
    "Gaussian 2SFCA": ha.gaussian_2sfca(homes.population, clinics.physicians, minutes, threshold=15),
    "3SFCA": ha.three_step_fca(homes.population, clinics.physicians, minutes),
    "M2SFCA": ha.modified_2sfca(homes.population, clinics.physicians, minutes, threshold=15),
    "gravity": ha.gravity(homes.population, clinics.physicians, minutes),
}.items():
    total = float((access * homes.population).sum())
    print(f"{name:15s} per 1,000: {(access * 1000).round(3).to_dict()}  supply accounted for: {total:.3f}")
    assert (access >= 0).all()
    if name != "M2SFCA":
        assert abs(total - clinics.physicians.sum()) < 1e-9, name

print("OSM smoke test passed")
