# tasks.py

import xarray as xr
import numpy as np

def filter_dod_any_year(input_dataset, output_dataset, dod_var="DOD", threshold=60):
    """
    Create a new dataset where pixels are removed if any year has DoD < threshold.
    
    Parameters:
    - input_dataset: Path to original dataset (final_dataset.nc)
    - output_dataset: Path to save the filtered dataset (final_dataset2.nc)
    - dod_var: Variable name for DoD
    - threshold: DoD threshold for filtering
    """
    
    print(f"Loading dataset: {input_dataset}")
    ds = xr.open_dataset(input_dataset)
    
    if dod_var not in ds:
        raise ValueError(f"Variable '{dod_var}' not found in dataset.")
    
    dod_data = ds[dod_var]  # shape: (year, pixel)
    
    print(f"Initial dataset shape: {ds.sizes}")
    
    # Identify pixels where ANY year has DoD < threshold
    mask_keep_pixels = (dod_data >= threshold).all(dim="year")  # shape: (pixel,)

    print(f"Pixels before filtering: {len(mask_keep_pixels)}")
    print(f"Pixels retained after filtering: {mask_keep_pixels.sum().item()}")

    # Apply filtering to all variables
    ds_filtered = ds.isel(pixel=mask_keep_pixels)

    print(f"Filtered dataset shape: {ds_filtered.sizes}")

    # Save new dataset
    ds_filtered.to_netcdf(output_dataset)
    print(f"Filtered dataset saved to: {output_dataset}")

if __name__ == "__main__":
    input_path = "/Users/yashnilmohanty/Desktop/final_dataset.nc"
    output_path = "/Users/yashnilmohanty/Desktop/final_dataset2.nc"
    
    filter_dod_any_year(input_path, output_path)

# final dataset size (year=15, pixel=19180)