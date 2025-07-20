#!/usr/bin/env python3
"""
merge_fpar_into_fireml.py  – one‑click edition
----------------------------------------------
Adds Peak_FPAR, Spring_FPAR, and Winter_FPAR from *FPAR_predictors.nc*
to *final_dataset5.nc* and saves the merged file as *final_dataset6.nc*.

Edit the three paths below if your files live elsewhere, then run.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import xarray as xr

# ─── EDIT THESE THREE PATHS ──────────────────────────────────────────
FILE_FIRE = Path("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
FILE_FPAR = Path("/Users/yashnilmohanty/Desktop/data/FPAR_predictors.nc")
FILE_OUT  = Path("/Users/yashnilmohanty/Desktop/final_dataset6.nc")
# ─────────────────────────────────────────────────────────────────────

FPAR_VARS = ["Peak_FPAR", "Spring_FPAR", "Winter_FPAR"]


def subset_fpar(da: xr.DataArray, idx: np.ndarray) -> xr.DataArray:
    """Return da(Time,total_length) → da(year,pixel) via integer index array."""
    valid = (idx >= 0) & (idx < da.sizes["total_length"])
    full = np.full((da.sizes["Time"], idx.size), np.nan, dtype=np.float32)
    if valid.any():
        sel = da.isel(total_length=xr.DataArray(idx[valid], dims="pixel_valid"))
        full[:, valid] = sel.values
    return xr.DataArray(full, dims=("year", "pixel"))


def main() -> None:
    if not FILE_FIRE.is_file() or not FILE_FPAR.is_file():
        sys.exit("❌  Check FILE_FIRE and FILE_FPAR paths – file(s) not found.")

    print("→ loading datasets …")
    ds_fire = xr.open_dataset(FILE_FIRE, engine="netcdf4")
    ds_fpar = xr.open_dataset(FILE_FPAR, engine="netcdf4")

    # pixel → total_length (1‑based → 0‑based)
    idx = ds_fire["ncoords_vector"].values.astype(int) - 1

    print("→ merging FPAR predictors …")
    for var in FPAR_VARS:
        if var not in ds_fpar:
            print(f"  ⚠︎  {var} missing – skipped")
            continue
        ds_fire[var] = subset_fpar(ds_fpar[var], idx)
        ds_fire[var].attrs.update(ds_fpar[var].attrs)
        print(f"  ✓  added {var}")

    print(f"→ writing merged file to {FILE_OUT} …")
    ds_fire.to_netcdf(FILE_OUT, engine="netcdf4")
    print("✓ done.")


if __name__ == "__main__":
    main()
