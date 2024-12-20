
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load the NetCDF data
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
ds = xr.open_dataset(data_path)

# Select the feature and compute the 15-year average
feature = 'aorcSpringShortwave'
feature_avg = ds[feature].mean(dim='nyears').values  # Average across 15 years

# Reshape the data (assuming a regular grid)
lat = ds['lat'].isel(year=0).values  # Extract latitude for the first year
lon = ds['lon'].isel(year=0).values  # Extract longitude for the first year

print("Feature data shape:", feature_avg.shape)
print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)
print("Number of ncoords in feature data:", ds.dims['ncoords'])

# Ensure lat and lon match the feature data
if len(feature_avg) != len(lat) or len(feature_avg) != len(lon):
    raise ValueError("Coordinate dimensions do not match the data dimensions.")

# Option 1: Plot as a Scatter Plot for Irregular Grids
plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=feature_avg, cmap='viridis', s=1)
plt.colorbar(label='15-Year Average of Spring Shortwave Radiation (W/m²)')
plt.title(f'15-Year Average Spatial Scatter Plot of {feature}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()