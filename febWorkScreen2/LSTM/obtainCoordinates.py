import xarray as xr
import numpy as np

# Load final_dataset2
ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset2.nc")

# Extract the lat/lon arrays (each shape = (pixel,))
lat_1d = ds["latitude"].values  # or ds["lat"].values, depending on variable naming
lon_1d = ds["longitude"].values # or ds["lon"].values

# Combine into shape (pixel, 2)
coords = np.column_stack([lat_1d, lon_1d])

# Now coords[i] is [lat, lon] for the i-th pixel
print("coords shape:", coords.shape)
