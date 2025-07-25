import xarray as xr
import numpy as np

# Load the NetCDF file
ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

# Extract VegTyp data (it's 2D: year x pixel)
veg_data = ds["VegTyp"].values

# Flatten the array and filter out NaNs
veg_flat = veg_data.flatten()
veg_clean = veg_flat[~np.isnan(veg_flat)].astype(int)

# Get unique vegetation type codes
unique_veg_types = np.unique(veg_clean)

# Print the list
print("Unique vegetation types used in VegTyp:")
print(unique_veg_types)
