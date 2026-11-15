"""Leakage test: features for a season must not change when future seasons are removed from the data.
Run from the repo root:  python -m pytest tests  (or: python tests/test_no_leakage.py)
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, build_features, load_tables  # noqa: E402


def test_features_ignore_the_future(season: int = 2024):
    t = load_tables(str(ROOT / "data" / "raw") + "/")
    full = build_features(t)
    keep = t["races"][t["races"].year <= season].raceId
    cut = {k: (v[v.raceId.isin(keep)] if "raceId" in v.columns else v) for k, v in t.items()}
    part = build_features(cut)
    a = full[full.year == season].set_index(["raceId", "driverId"])[FEATURES].sort_index().astype("float64")
    b = part[part.year == season].set_index(["raceId", "driverId"])[FEATURES].sort_index().astype("float64")
    assert a.shape == b.shape
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


if __name__ == "__main__":
    test_features_ignore_the_future()
    print("OK: no future information leaks into the features")
