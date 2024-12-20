
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load the NetCDF data
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
ds = xr.open_dataset(data_path)

# Select the feature: Elevation
feature = 'Elevation'
elevation_data = ds[feature].values  # Extract elevation data

# Check if elevation has a time dimension (nyears)
if 'year' in ds[feature].dims:
    # Compute the average elevation over time (nyears) if it exists
    elevation_avg = ds[feature].mean(dim='year').values
else:
    # Elevation might already be static (no time dimension)
    elevation_avg = elevation_data

# Extract static lat and lon (use the first year or compute average if time-dependent)
lat = ds['lat'].isel(year=0).values  # Use the first year if lat is time-dependent
lon = ds['lon'].isel(year=0).values  # Use the first year if lon is time-dependent

# Debugging shapes
print("Elevation data shape:", elevation_avg.shape)
print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)

# Ensure lat and lon match the elevation data
if len(elevation_avg) != len(lat) or len(elevation_avg) != len(lon):
    raise ValueError("Coordinate dimensions do not match the data dimensions.")

# Option 1: Scatter Plot for Irregular Grids
plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=elevation_avg, cmap='terrain', s=1)
plt.colorbar(label='Elevation (meters MSL)')
plt.title('Spatial Scatter Plot of Elevation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('Elevation_spatial_scatter.png', dpi=300)
plt.show()
