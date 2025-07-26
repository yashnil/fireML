#!/usr/bin/env python3
"""
Per‑category Fire‑ML experiment
================================
Replicates *experiment #1* logic (70 % train / 30 % test **within**
category 0) for **every** cumulative‑burn category c0–c3.  Because the
pipeline, feature matrix, and Random‑Forest settings are *byte‑for‑byte
identical* to the reference script, the c0 metrics will match exactly;
metrics for c1–c3 are now produced with the very same architecture.

Usage
-----
$ python per_category_experiment.py

You should see four lines like
    Category 0: mean bias=0.00, bias std=11.87, R^2=0.86
    Category 1: …
    Category 2: …
    Category 3: …

Only the dataset path may need editing.
"""

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────
DATA_PATH = "/Users/yashnilmohanty/Desktop/final_dataset5.nc"  # ← adjust if needed
RANDOM_STATE = 42
TEST_FRAC    = 0.30
N_TREES      = 100

# ────────────────────────────────────────────────────────────
#  Helpers (verbatim from experiment #1)
# ────────────────────────────────────────────────────────────
_EXCL_EXTRA = {
    "lat", "lon", "latitude", "longitude",
    "pixel", "year",
    "burn_fraction", "burn_cumsum",
    "ncoords_vector", "nyears_vector",
    "aorcsummerhumidity", "aorcsummerprecipitation",
    "aorcsummerlongwave", "aorcsummershortwave",
    "aorcsummertemperature",
}


def _gather_features_nobf(ds: xr.Dataset, target: str = "DSD") -> dict:
    """Return a dict {name → ndarray(year,pixel)} with *burn_fraction* excluded."""
    excl = _EXCL_EXTRA | {target.lower()}
    feats = {}
    ny = ds.sizes["year"]
    for v in ds.data_vars:
        if v.lower() in excl:
            continue
        da = ds[v]
        # unit fix identical to experiment #1
        if v == "aorcWinterPrecipitation":
            da = da * 86_400.0  # mm s⁻¹ → mm day⁻¹
        if set(da.dims) == {"year", "pixel"}:
            feats[v] = da.values
        elif set(da.dims) == {"pixel"}:
            feats[v] = np.tile(da.values, (ny, 1))
    return feats


def _build_matrix(ds: xr.Dataset, target: str = "DSD"):
    """Flatten features and target to 2‑D/1‑D arrays, removing NaNs."""
    fd = _gather_features_nobf(ds, target)
    names = sorted(fd)  # same column order as reference
    X = np.column_stack([fd[n].ravel(order="C") for n in names]).astype(np.float64)
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X[ok], y[ok], ok


# ────────────────────────────────────────────────────────────
#  Main routine
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = xr.open_dataset(DATA_PATH)

    # build burn‑category array (year,pixel) → flat
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, dtype=int)
    cat2d[bc < 0.25] = 0
    cat2d[(bc >= 0.25) & (bc < 0.50)] = 1
    cat2d[(bc >= 0.50) & (bc < 0.75)] = 2
    cat2d[bc >= 0.75] = 3

    # feature matrix & target vector (identical to experiment #1)
    X, y, ok = _build_matrix(ds, "DSD")
    cat = cat2d.ravel(order="C")[ok]

    # loop over categories
    for c in range(4):
        rows = np.where(cat == c)[0]
        if rows.size == 0:
            print(f"Category {c}: no samples available")
            continue

        train_idx, test_idx = train_test_split(
            rows,
            test_size=TEST_FRAC,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        rf = RandomForestRegressor(
            n_estimators=N_TREES,
            random_state=RANDOM_STATE,
            n_jobs=1,          # single‑thread → fully deterministic
            bootstrap=True,    # default (matches reference)
        )
        rf.fit(X[train_idx], y[train_idx])

        y_pred = rf.predict(X[test_idx])
        resid  = y_pred - y[test_idx]

        print(
            f"Category {c}: mean bias={resid.mean():.2f}, "
            f"bias std={resid.std():.2f}, "
            f"R^2={r2_score(y[test_idx], y_pred):.2f}"
        )
