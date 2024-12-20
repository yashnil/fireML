import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Open the NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/data/AORC_Data_Northen_CA/AORC_Data_Northen_CA_Winter_CY2011.nc'
dataset = nc.Dataset(file_path, mode='r')

for var in dataset.variables:
    print(var)
    
# Extract latitude, longitude, and RAINRATE
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
rainrate = dataset.variables['RAINRATE'][:]  # Shape (91, 95324)

# Close the dataset
dataset.close()

# Select a specific time point, e.g., the first time point (index 0)
rainrate_slice = rainrate[0, :]  # Extract data for the first time point

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=rainrate_slice, cmap='Blues', s=10, marker='o')  # Use scatter plot for unstructured grid
plt.colorbar(label='Rain Rate (mm/hr)')
plt.title('Rain Rate in Northern California - Fall 1981 (Time Point 0)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
