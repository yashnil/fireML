#!/usr/bin/env python3
"""
rf_burned_stratified.py  (v3 – NICE_NAME overhaul)
===================================================
Random‑Forest robustness test on the *burned* categories (c1–c3):
* 70 % stratified train / 30 % test inside c1–c3.
* Prints per‑category R², mean bias, bias SD.
* Shows a bar plot of the **Top‑10** feature importances with
  human‑readable axis labels.

Data‑cleansing rules
--------------------
* Drop every predictor that starts with ``aorcSummer``.
* Drop ``ncoords_vector`` and ``nyears_vector``.
* Convert ``aorcWinterPrecipitation`` from mm s⁻¹ → mm day⁻¹.

Pretty display names (``NICE_NAME``)
------------------------------------
For all seasonal AORC variables the leading "aorc" is stripped and the
season + parameter is shown instead, e.g. ``aorcSpringTemperature`` →
"Spring Temperature".  Shortwave variables get a ↓ arrow.  Additional
static layers (e.g. *Elevation*, *VegTyp*) have bespoke labels.

Usage
-----
```bash
python rf_burned_stratified.py /path/to/final_dataset5.nc
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
#  NICE_NAME mapping  (seasonal + static layers)
# ────────────────────────────────────────────────────────────────
NICE_NAME: Dict[str, str] = {}
for _season in ("Fall", "Winter", "Spring", "Summer"):
    for _feat in ("Temperature", "Precipitation", "Humidity", "Shortwave", "Longwave"):
        key = f"aorc{_season}{_feat}"
        arrow = "↓" if _feat == "Shortwave" else ""
        NICE_NAME[key] = f"{_season} {_feat}{arrow}"

# bespoke layers
NICE_NAME.update(
    {
        "peakValue": "Peak SWE (mm)",
        "Elevation": "Elevation (m)",
        "slope": "Slope",
        "aspect_ratio": "Aspect Ratio",
        "VegTyp": "Vegetation Type",
        "sweWinter": "Winter SWE",
    }
)

# ────────────────────────────────────────────────────────────────
#  Feature‑matrix helpers  (burn_fraction & unwanted vars excluded)
# ────────────────────────────────────────────────────────────────

def gather_features_nobf(ds: xr.Dataset, target: str = "DOD") -> Dict[str, np.ndarray]:
    """Collect predictors → {name: ndarray(year,pixel)}."""

    exclude_exact = {
        target.lower(),
        "lat",
        "lon",
        "latitude",
        "longitude",
        "pixel",
        "year",
        "burn_fraction",
        "burn_cumsum",
        "ncoords_vector",
        "nyears_vector",
    }

    feats: Dict[str, np.ndarray] = {}
    ny = ds.sizes["year"]

    for v in ds.data_vars:
        if v.lower() in exclude_exact:
            continue
        if v.lower().startswith("aorcsummer"):
            continue  # drop ALL summer variables

        da = ds[v]

        # mm s⁻¹ → mm day⁻¹ for winter precipitation
        if v == "aorcWinterPrecipitation":
            da = da * 86_400.0

        if set(da.dims) == {"year", "pixel"}:
            feats[v] = da.values
        elif set(da.dims) == {"pixel"}:  # static → repeat over years
            feats[v] = np.tile(da.values, (ny, 1))

    return feats


def flatten_nobf(ds: xr.Dataset, target: str = "DOD") -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok


# ────────────────────────────────────────────────────────────────
#  Plot helper
# ────────────────────────────────────────────────────────────────

def plot_top10_features(rf: RandomForestRegressor, feature_names: List[str]) -> None:
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8, 4))
    plt.bar(range(10), imp[idx])
    plt.xticks(
        range(10),
        [NICE_NAME.get(feature_names[i], feature_names[i]) for i in idx],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Importance")
    plt.title("Top‑10 Predictor Importances (RF)")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────
#  Core routine
# ────────────────────────────────────────────────────────────────

def rf_burned_stratified(ds: xr.Dataset) -> None:
    """70/30 stratified split on c1–c3 and RF evaluation."""

    # 1. Burn categories ------------------------------------------------------
    bc = ds["burn_cumsum"].values
    cat = np.full_like(bc, -1, dtype=int)
    cat[bc < 0.25] = 0
    cat[(bc >= 0.25) & (bc < 0.50)] = 1
    cat[(bc >= 0.50) & (bc < 0.75)] = 2
    cat[bc >= 0.75] = 3

    # 2. Feature matrix -------------------------------------------------------
    X_flat, y_flat, feat_names, ok = flatten_nobf(ds, target="DOD")

    X_valid, y_valid = X_flat[ok], y_flat[ok]
    cat_flat = cat.ravel(order="C")[ok]

    # 3. Burned‑only subset ---------------------------------------------------
    burned = cat_flat >= 1
    X_burn, y_burn, cat_burn = X_valid[burned], y_valid[burned], cat_flat[burned]

    # 4. Stratified 70/30 split ----------------------------------------------
    X_tr, X_te, y_tr, y_te, cat_tr, cat_te = train_test_split(
        X_burn,
        y_burn,
        cat_burn,
        test_size=0.30,
        random_state=42,
        stratify=cat_burn,
    )

    # 5. Train RF -------------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    # 6. Metrics --------------------------------------------------------------
    print("Per‑category metrics (TEST 30 % of c1–c3):\n")
    for c in (1, 2, 3):
        sel = cat_te == c
        if not sel.any():
            print(f"  cat{c}: N=0")
            continue
        bias = y_pred[sel] - y_te[sel]
        r2 = r2_score(y_te[sel], y_pred[sel])
        print(
            f"  cat{c}: N={sel.sum():4d}  R²={r2:6.3f}  "
            f"Mean Bias={bias.mean():7.2f}  Bias SD={bias.std(ddof=0):6.2f}"
        )

    # 7. Feature‑importance plot ---------------------------------------------
    plot_top10_features(rf, feat_names)


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="RF robustness test on burned categories (c1–c3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "dataset",
        nargs="?",
        default="/Users/yashnilmohanty/Desktop/final_dataset5.nc",
        help="Path to the NetCDF file",
    )
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        sys.exit(f"ERROR: file not found → {ds_path}")

    ds = xr.open_dataset(ds_path)
    rf_burned_stratified(ds)


if __name__ == "__main__":
    main()
