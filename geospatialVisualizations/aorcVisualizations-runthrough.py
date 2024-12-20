import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Open the NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/data/AORC_Data_Northen_CA/AORC_Data_Northen_CA_Fall_CY1980.nc'
dataset = nc.Dataset(file_path, mode='r')


# Extract latitude, longitude, and T2D (2-meter temperature)
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
t2d = dataset.variables['LWDOWN'][:]  # Shape (91, 95324)

print(t2d.shape)
# Close the dataset
dataset.close()

# Select a specific time point, e.g., the first time point (index 0)
t2d_slice = t2d[0, :]  # Extract data for the first time point

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=t2d_slice, cmap='coolwarm', s=10, marker='o')  # Use scatter plot for unstructured grid
plt.colorbar(label='Temperature (K)')
plt.title('2-Meter Temperature in Northern California - Fall 1980 (Time Point 0)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
