# Code intended to verify that Geo2D and Mateorological Data have the same spatial resolution

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the NetCDF dataset
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
ds = xr.open_dataset(data_path)

# Extract coordinates if available
# Check if lat/lon are stored separately or associated with `npixels`
if 'lat' in ds.coords and 'lon' in ds.coords:
    lat = ds['lat'].values
    lon = ds['lon'].values
    print("Latitude and Longitude are directly available in the dataset.")
else:
    print("Latitude and Longitude coordinates are not explicitly provided.")

# Check shape and dimensions of `Geo2D` data (e.g., Elevation)
elevation = ds['Elevation']  # Geo2D variable
print("Elevation dimensions:", elevation.dims)
print("Elevation shape:", elevation.shape)

# If `npixels` corresponds to lat/lon points, check alignment
if 'npixels' in elevation.dims:
    npixels = elevation['npixels']
    print("npixels dimension detected. Shape:", npixels.shape)

    # Attempt to match npixels to lat/lon if possible
    if 'lat' in ds.coords and 'lon' in ds.coords:
        print(f"Lat/Lon shape: {lat.shape}, {lon.shape}")
        print(f"npixels shape: {npixels.shape}")
        # Plot a subset to visualize correspondence
        plt.figure(figsize=(8, 6))
        plt.scatter(lon, lat, s=1, label='Meteorological Grid Points')
        plt.title("Grid Points in Dataset")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()
    else:
        print("Lat/Lon not explicitly provided for npixels.")

# Check alignment between Meteorological and Geo2D data
# Example using Elevation with 2D meteorological variables
meteo_var = ds['aorcFallTemperature']  # Example 2D variable
if meteo_var.shape[-1] == elevation.shape[-1]:
    print("Geo2D and Meteorological data share the same spatial resolution!")
else:
    print("Geo2D and Meteorological data have different spatial resolutions.")

'''
# Visualize data if needed
plt.figure(figsize=(8, 6))
plt.imshow(elevation.isel(year=0).values.reshape(300, 318), cmap="viridis")  # Reshape based on metadata
plt.colorbar(label="Elevation (meters MSL)")
plt.title("Elevation Geo2D Data (Year 0)")
plt.show()

'''