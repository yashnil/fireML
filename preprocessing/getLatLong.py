import netCDF4 as nc
import numpy as np

# Path to the NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/data/Static_data_all.nc'

# Open the NetCDF file
dataset = nc.Dataset(file_path, mode='r')

# Access latitude and longitude arrays
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

# Check the shape of lat and lon arrays
lat_shape = lat.shape
lon_shape = lon.shape

print(f"Latitude array shape: {lat_shape}")
print(f"Longitude array shape: {lon_shape}")

# Initialize a list to store coordinates
coords = []

# Iterate over the indices of the lat array
for i in range(len(lat)):
    coords.append((lat[i], lon[i]))

dataset.close()
