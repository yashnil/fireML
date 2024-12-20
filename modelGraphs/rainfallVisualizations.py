
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load the NetCDF data
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
ds = xr.open_dataset(data_path)

# Select the feature: Rainfall (e.g., aorcFallRain)
feature = 'aorcFallRain'  # Replace with the desired rainfall variable
rainfall_data = ds[feature].values  # Extract rainfall data

# Compute the 15-year average across time (nyears)
rainfall_avg = ds[feature].mean(dim='nyears').values  # Average over time

# Extract static lat and lon (use the first year or compute average if time-dependent)
lat = ds['lat'].isel(year=0).values  # Use the first year if lat is time-dependent
lon = ds['lon'].isel(year=0).values  # Use the first year if lon is time-dependent

# Debugging shapes
print("Rainfall data shape:", rainfall_avg.shape)
print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)

# Ensure lat and lon match the rainfall data
if len(rainfall_avg) != len(lat) or len(rainfall_avg) != len(lon):
    raise ValueError("Coordinate dimensions do not match the data dimensions.")

# Option 1: Scatter Plot for Irregular Grids
plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=rainfall_avg, cmap='Blues', s=1)
plt.colorbar(label='15-Year Average Rainfall (mm)')
plt.title(f'Spatial Scatter Plot of {feature} (15-Year Average)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
