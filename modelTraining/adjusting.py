import xarray as xr
import numpy as np

# Load the NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'  # Update with your file path
data = xr.open_dataset(file_path)

# Adjust 'npixels' indexing to start from 1
if 'npixels' in data.dims:
    data = data.assign_coords(npixels=(data['npixels'] - 1))

# Adjust 'year' indexing to match 2004–2018
if 'year' in data.coords:
    data = data.assign_coords(year=(data['year'] - 2004))

# Save the modified data back to a new file
updated_file_path = '/Users/yashnilmohanty/Desktop/adjusted_combined_data.nc'  # Update output path
data.to_netcdf(updated_file_path)

print(f"Updated dataset saved to {updated_file_path}")
