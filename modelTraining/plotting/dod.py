import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the processed NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/preprocessed_combined_data.nc'  # Update with your actual file path
data = xr.open_dataset(file_path)

# Extract the relevant data
dod = data['DOD'].values  # Extract DOD values
lat = data['lat'].values  # Extract latitude values
lon = data['lon'].values  # Extract longitude values

# Identify non-NaN coordinates in `DOD`
valid_coords = ~np.isnan(dod)  # Boolean mask for valid coordinates

# Get the corresponding lat/lon and DOD values for valid coordinates
valid_lats = lat[valid_coords]
valid_lons = lon[valid_coords]
valid_dod = dod[valid_coords]

# Plot the valid data points
plt.figure(figsize=(10, 8))
sc = plt.scatter(valid_lons, valid_lats, c=valid_dod, cmap='viridis', s=20)
plt.colorbar(sc, label='Date of Disappearance (DOD)')
plt.title('DOD Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Show the plot
plt.show()
