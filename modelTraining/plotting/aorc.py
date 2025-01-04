import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the processed NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/preprocessed_combined_data.nc'  # Update with your actual file path
data = xr.open_dataset(file_path)

# Extract the relevant data
fall_longwave = data['aorcFallLongwave'].values  # Extract aorcFallLongwave values
lat = data['lat'].values                         # Extract latitude values
lon = data['lon'].values                         # Extract longitude values

# Identify non-NaN coordinates in `aorcFallLongwave`
valid_coords = ~np.isnan(fall_longwave)  # Boolean mask for valid coordinates

# Get the corresponding lat/lon and fall_longwave values for valid coordinates
valid_lats = lat[valid_coords]
valid_lons = lon[valid_coords]
valid_fall_longwave = fall_longwave[valid_coords]

# Plot the valid data points
plt.figure(figsize=(10, 8))
sc = plt.scatter(valid_lons, valid_lats, c=valid_fall_longwave, cmap='cividis', s=20)
plt.colorbar(sc, label='Fall Longwave Radiation (W/m²)')
plt.title('Fall Longwave Radiation Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Show the plot
plt.show()
