import xarray as xr
import os

def rename_variable_in_netcdf(file_path, old_var, new_var):
    # Open dataset
    ds = xr.open_dataset(file_path)

    # Check if the old variable exists
    if old_var not in ds.variables:
        print(f"{old_var} not found in {file_path}")
        return

    # Rename and save to a temporary file
    renamed_ds = ds.rename({old_var: new_var})
    temp_path = file_path.replace(".nc", "_temp.nc")
    renamed_ds.to_netcdf(temp_path)

    # Replace original file
    os.replace(temp_path, file_path)
    print(f"Renamed '{old_var}' to '{new_var}' in {file_path}")

# File paths
file1 = "/Users/yashnilmohanty/Desktop/final_dataset5.nc"
file2 = "/Users/yashnilmohanty/Desktop/final_dataset6.nc"

# Perform the renaming
rename_variable_in_netcdf(file1, "DOD", "DSD")
rename_variable_in_netcdf(file2, "DOD", "DSD")
