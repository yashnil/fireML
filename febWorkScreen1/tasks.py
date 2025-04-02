# tasks.py

import xarray as xr
import numpy as np
import glob
import os

def main():
    ########################
    # 1) COMBINE DATASETS
    ########################
    path_fire = "/Users/yashnilmohanty/Desktop/fire_modified.nc"
    path_adj  = "/Users/yashnilmohanty/Desktop/adjusted_combined_data.nc"

    ds_fire = xr.open_dataset(path_fire)
    ds_adj  = xr.open_dataset(path_adj)

    # Force both to have year=range(15)
    ds_fire = ds_fire.assign_coords(year=range(15))
    ds_adj  = ds_adj.assign_coords(year=range(15))

    # Rename dims so they match
    ds_fire = ds_fire.rename({"year": "year", "location": "pixel"})
    ds_adj  = ds_adj.rename({"nyears": "year", "ncoords": "pixel"})
    ds_adj  = ds_adj.rename({"npixels": "pixel"})

    ds_combined = xr.merge([ds_fire, ds_adj], compat='override')
    print("After combine, ds_combined variables:", list(ds_combined.data_vars))

    ########################
    # 2) FILTER BY DoD & Compute peak SWE + DOP
    ########################

    DOD_array = ds_combined["DOD"]  # shape: (year=15, pixel=95324)

    # --- A) Remove pixels with ANY NaNs in DOD timeseries
    dod_nan_mask = ~np.isnan(DOD_array).any(dim="year")  
    ds_step1 = ds_combined.isel(pixel=dod_nan_mask)
    print(f"Shape after removing DOD NaNs: {ds_step1.sizes}")

    # --- B) Remove pixels whose mean(DOD) < 60
    DOD_threshold = 60
    mean_dod = ds_step1["DOD"].mean(dim="year")  
    valid_dod_range_mask = (mean_dod >= DOD_threshold)
    ds_step2 = ds_step1.isel(pixel=valid_dod_range_mask)
    print(f"Shape after removing mean(DOD) < {DOD_threshold}: {ds_step2.sizes}")

    # --- C) Compute annual max SWE (peakValue) + DOP from SNODAS
    path_snodas = "/Users/yashnilmohanty/Desktop/data/SNODAS_SWE_by_WY/SNODAS_SWE_*.nc"
    files_snodas = sorted(glob.glob(path_snodas))  

    peak_all_years = []
    dop_all_years = []

    for i, f in enumerate(files_snodas):
        ds_sno = xr.open_dataset(f)
        swe_daily = ds_sno["SWE"]  # shape: (Time, 95324)

        # Compute peak SWE (Max SWE per pixel)
        peakValue = swe_daily.max(dim="Time")  # shape: (95324,)

        # Find the first day (DOP) where SWE reaches the peak
        dop_all = []
        for time in swe_daily["Time"]:
            is_peak_day = swe_daily.sel(Time=time) == peakValue
            dop_all.append(is_peak_day * time)

        DOP = xr.concat(dop_all, dim="Time").max(dim="Time")
        DOP = DOP.where(~np.isnan(peakValue), np.nan)

        # Expand dims to match (year, pixel)
        peakValue = peakValue.expand_dims(dim={"year": [i]})
        DOP = DOP.expand_dims(dim={"year": [i]})

        peak_all_years.append(peakValue)
        dop_all_years.append(DOP)

    peakVal_da = xr.concat(peak_all_years, dim="year").rename({"total_length": "pixel"})
    dop_da     = xr.concat(dop_all_years, dim="year").rename({"total_length": "pixel"})

    # Compute mean peak SWE
    peakSWE_mean_da = peakVal_da.mean(dim="year")
    swe_threshold = 50
    valid_swe_mask = (peakSWE_mean_da >= swe_threshold)

    # apply to ds_step2
    kept_pixels_step2 = ds_step2.pixel.values  
    mask_swe_for_step2 = np.array([valid_swe_mask[pix] for pix in kept_pixels_step2], dtype=bool)
    ds_step3 = ds_step2.isel(pixel=mask_swe_for_step2)
    print(f"Shape after removing avg peak SWE < {swe_threshold}: {ds_step3.sizes}")

    ########################
    # 3) Assign Computed peakValue and DOP
    ########################

    # Subset peakVal_da, dop_da
    pixel_idx = ds_step3["pixel"].values.astype(int)
    year_idx  = ds_step3["year"].values.astype(int)

    # DEBUG 1: Print first 50 pixel indices in ds_step3
    print("First 50 pixel IDs in ds_step3:", pixel_idx[:50])

    # DEBUG 2: Check if pixel ID=45 is in pixel_idx
    if 45 in pixel_idx:
        pos_index = np.where(pixel_idx == 45)[0]
        print(f"Pixel ID=45 found at position(s) {pos_index} in ds_step3.")
    else:
        print("Pixel ID=45 is NOT present in ds_step3 pixel array. => Explains any mismatch.")

    # DEBUG 3: Print sample of peakVal_da and dop_da
    print("peakVal_da sample, isel(pixel=[30,31,32,45], year=0):")
    try:
        sample_peak = peakVal_da.isel(pixel=[30,31,32,45], year=0).values
        print("peakVal (year=0) for pixel rows [30,31,32,45]:", sample_peak)
    except Exception as e:
        print("Error sampling peakVal_da for pixel rows [30,31,32,45]:", e)

    print("dop_da sample, isel(pixel=[30,31,32,45], year=0):")
    try:
        sample_dop = dop_da.isel(pixel=[30,31,32,45], year=0).values
        print("dop (year=0) for pixel rows [30,31,32,45]:", sample_dop)
    except Exception as e:
        print("Error sampling dop_da for pixel rows [30,31,32,45]:", e)

    final_peakVal_da = peakVal_da.isel(pixel=pixel_idx-1, year=year_idx)
    final_dop_da     = dop_da.isel(pixel=pixel_idx-1, year=year_idx)

    final_peakVal_da.name = "peakValue"
    final_dop_da.name     = "DOP"

    ds_step3["peakValue"] = final_peakVal_da
    ds_step3["DOP"]       = final_dop_da

    print("Successfully computed peakValue + DOP from daily SNODAS data.")

    # Possibly drop sweSpring
    if "sweSpring" in ds_step3.variables:
        ds_step3 = ds_step3.drop_vars("sweSpring")

    # Drop dims
    ds_step3 = ds_step3.drop_dims("npixels", errors="ignore")
    ds_step3 = ds_step3.drop_dims("lat", errors="ignore")
    ds_step3 = ds_step3.drop_dims("lon", errors="ignore")

    out_path = "/Users/yashnilmohanty/Desktop/final_dataset.nc"
    ds_step3.to_netcdf(out_path)
    print(f"Final dataset saved to {out_path}")

    print("Final dataset dims:", ds_step3.sizes)
    print("Variables in final dataset:", list(ds_step3.data_vars))

if __name__ == "__main__":
    main()

# final dataset size (year=15, pixel=40894)