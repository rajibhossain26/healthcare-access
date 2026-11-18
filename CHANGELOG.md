# Changelog

## 0.2.1 (2026-11-18)

- Fix: `equity.share_below` and `equity.group_summary` now leave out locations whose access value is missing. Before, those people were still counted in the population, which pulled the results down.
- The `osm` extra now also installs scikit-learn, which OSMnx needs to snap points to an unprojected road network.
- Validation: loop-based reference implementations of the published equations, cross-checks against PySAL `access` and Aceso (32 of 32 agree to floating-point precision), and an OpenStreetMap smoke test.

## 0.2.0 (2026-11-18)

- New models: `three_step_fca` (Wan, Zou & Sternberg 2012) and `modified_2sfca` (Delamater 2013).
- Packaged for PyPI as `healthcare-access` (the import name stays `healthaccess`).
- Release workflow that builds, checks and publishes to PyPI with trusted publishing.
- Citation metadata (`CITATION.cff`).

## 0.1.0

- 2SFCA, Enhanced 2SFCA, Gaussian 2SFCA, gravity model and a generic `fca` with any decay function.
- Haversine, Euclidean, network and OpenStreetMap travel-cost helpers.
- Equity summaries: weighted Gini, share below a threshold, group comparisons.
