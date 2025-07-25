#!/usr/bin/env python3
"""Filter out unwanted vegetation classes from a pixel‑wise NetCDF dataset.

Creates *final_dataset5.nc* that is identical to *final_dataset4.nc* except
that pixels whose vegetation type is Water (16), Barren (19) or Bare Ground
Tundra (23) have been removed across **all** variables.

Run:
    python filter_vegtypes.py  [--in path/to/final_dataset4.nc]
                               [--out path/to/final_dataset5.nc]

If the output path is the same as the input path you will overwrite the
original file.  Use with caution.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

# vegetation classes to remove
BAD_VEG = {16, 19, 23}

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Remove specified vegetation types from a Fire‑ML pixel dataset.")
    p.add_argument("--in", dest="infile",  default="/Users/yashnilmohanty/Desktop/final_dataset4.nc",
                   help="Path to the input NetCDF file (default: final_dataset4.nc)")
    p.add_argument("--out", dest="outfile", default="/Users/yashnilmohanty/Desktop/final_dataset5.nc",
                   help="Path for the filtered NetCDF file (default: final_dataset5.nc)")
    return p

def main() -> None:
    args = build_arg_parser().parse_args()
    infile  = Path(args.infile).expanduser().resolve()
    outfile = Path(args.outfile).expanduser().resolve()

    print(f"Reading  {infile} …")
    ds = xr.open_dataset(infile)

    if "VegTyp" not in ds:
        raise KeyError("Dataset has no variable 'VegTyp'.")

    veg = ds["VegTyp"].values  # dims: ('pixel',) or ('year','pixel')

    # treat VegTyp as time‑invariant per pixel; if it has a 'year' dim, take the
    # first year's slice (they are identical when vegetation is static).
    if veg.ndim == 2:
        veg_static = veg[0, :]
    else:
        veg_static = veg

    # build boolean mask: True for pixels we *keep*
    keep_mask = ~np.isin(veg_static, list(BAD_VEG))
    n_total   = keep_mask.size
    n_keep    = int(keep_mask.sum())
    n_drop    = n_total - n_keep

    print(f"Pixels before filtering : {n_total}")
    print(f"Pixels removed          : {n_drop} (classes {sorted(BAD_VEG)})")
    print(f"Pixels after filtering  : {n_keep}")

    # apply mask along the 'pixel' dimension for every variable & coordinate
    ds_filtered = ds.isel(pixel=keep_mask)

    print(f"Writing  {outfile} …")
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_filtered.data_vars}
    ds_filtered.to_netcdf(outfile, encoding=encoding)
    print("Done.")

if __name__ == "__main__":
    main()
