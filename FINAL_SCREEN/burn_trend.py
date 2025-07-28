#!/usr/bin/env python3
"""
Compute annual total burned area (acres) for northern California and test
**whether the trend is significantly increasing** using a *one‑tailed* linear
regression (H₀: slope ≤ 0, H₁: slope > 0).

Revision – 2025‑06‑30
---------------------
* X‑axis labels shifted by +2004 so NetCDF year 0 → 2004, 1 → 2005, …
* Added one‑tailed p‑value: `p_one = p_two/2` when the fitted slope is
  positive (otherwise `p_one = 1 – p_two/2`).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Geod
from scipy.stats import linregress

DATA_PATH = Path("/Users/yashnilmohanty/Desktop/final_dataset4.nc")
if not DATA_PATH.exists():
    sys.exit(f"❌ {DATA_PATH} not found. Place the NetCDF file in this directory.")

ds = xr.open_dataset(DATA_PATH)

# ----------------------------------------------------------------------------
# 1 · Spatial mask for northern California
# ----------------------------------------------------------------------------
lat = ds["latitude"].values.ravel()
lon = ds["longitude"].values.ravel()
mask_ca = (
    (lat >= 37.0) & (lat <= 42.5) &
    (lon >= -124.5) & (lon <= -117.5)
)

# ----------------------------------------------------------------------------
# 2 · Pixel area (m²)
# ----------------------------------------------------------------------------
if "pixel_area" in ds:
    area_m2 = ds["pixel_area"].values.ravel()
else:
    try:
        geod = Geod(ellps="WGS84")
        dlat = np.abs(np.diff(lat.reshape(-1, 1), axis=0)).mean() if lat.size > 1 else 0.02
        dlon = np.abs(np.diff(lon.reshape(-1, 1), axis=0)).mean() if lon.size > 1 else 0.02
        area_m2 = np.array([
            abs(geod.polygon_area_perimeter(
                [lo - dlon/2, lo + dlon/2, lo + dlon/2, lo - dlon/2],
                [la - dlat/2, la - dlat/2, la + dlat/2, la + dlat/2]
            )[0]) for la, lo in zip(lat, lon)])
    except Exception as e:
        print("Could not derive cell areas (", e, ") – using 1 km² per cell.")
        area_m2 = np.full_like(lat, 1_000_000.0)

# ----------------------------------------------------------------------------
# 3 · Annual burned area (acres)
# ----------------------------------------------------------------------------
burn_frac = ds["burn_fraction"].values               # (year, pixel)
pix_idx   = np.where(mask_ca)[0]
area_ca   = area_m2[pix_idx]
area_ca2d = np.tile(area_ca, (burn_frac.shape[0], 1))

burn_area_m2 = (burn_frac[:, pix_idx] * area_ca2d).sum(axis=1)
burn_area_ac = burn_area_m2 / 4046.8564224            # m² → acres

# ----------------------------------------------------------------------------
# 4 · Year vector for display
# ----------------------------------------------------------------------------
raw_years = ds["year"].values                         # 0,1,…
years     = raw_years + 2003                         # calendar years
series    = pd.Series(burn_area_ac, index=years, name="burn_acres")

# ----------------------------------------------------------------------------
# 5 · Linear regression (two‑tailed) → convert to one‑tailed
# ----------------------------------------------------------------------------
res = linregress(raw_years, series.values)            # slope, intercept, r, p_two, stderr
slope, intercept, r, p_two, std_err = res
p_one = p_two/2 if slope > 0 else 1 - p_two/2         # one‑tailed (increase only)
trend_line = intercept + slope * raw_years

print(f"Slope = {slope:.1f} ± {std_err:.1f} acres/yr  (one‑tailed p = {p_one:.3g})")
print(f"Period: {years[0]}–{years[-1]}")

# ----------------------------------------------------------------------------
# 6 · Plot
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(years, series.values, label="Annual burned area", alpha=0.7)
ax.plot(years, trend_line, lw=2,
        label=f"Trend: {slope:.0f} ac/yr (p = {p_one:.3g})")
# ax.set_xlabel("Year")
ax.set_ylabel("Burned area (acres)")
ax.set_xlim(2002.5, 2017.5)
ax.set_xticks(np.arange(years[0], years[-1] + 1, 1))
ax.set_title("Burn Area Trend in Northern California")
ax.legend()
fig.tight_layout()
fig.savefig("burn_area_trend.png", dpi=300)
print("✓ Figure saved to burn_area_trend.png")
