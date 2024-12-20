import numpy as np
import xarray as xr
import pandas as pd

# Open the NetCDF file in append mode
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
with xr.open_dataset(data_path, mode='a') as ds:
    # Debug: Check the number of NaN values in SWE Winter
    print("Initial NaN count in SWE Winter:", np.isnan(ds['sweWinter'].values).sum())

    # --- OPTION 1: Replace NaN with Mean ---
    sweWinter_mean = np.nanmean(ds['sweWinter'].values)
    ds['sweWinter'].values = np.nan_to_num(ds['sweWinter'].values, nan=sweWinter_mean)
    print("After mean imputation, NaN count in SWE Winter:", np.isnan(ds['sweWinter'].values).sum())

    # --- OPTION 2: Replace NaN with Spatial Interpolation ---
    # Uncomment to use spatial interpolation
    # for year in range(ds.dims['nyears']):
    #     swe_data = ds['sweWinter'].isel(nyears=year).values
    #     ds['sweWinter'][year, :] = pd.Series(swe_data).interpolate(limit_direction='both').values
    # print("After spatial interpolation, NaN count in SWE Winter:", np.isnan(ds['sweWinter'].values).sum())

    # --- OPTION 3: Replace NaN with Temporal Interpolation ---
    # Uncomment to use temporal interpolation
    # for coord in range(ds.dims['ncoords']):
    #     swe_data = ds['sweWinter'][:, coord].values
    #     ds['sweWinter'][:, coord] = pd.Series(swe_data).interpolate(limit_direction='both').values
    # print("After temporal interpolation, NaN count in SWE Winter:", np.isnan(ds['sweWinter'].values).sum())

    # Debug: Ensure no NaN values remain in SWE Winter
    print("Final NaN count in SWE Winter:", np.isnan(ds['sweWinter'].values).sum())

    # The changes will be saved directly to the file upon exiting the 'with' block
    print("Changes saved directly to the original NetCDF file.")
