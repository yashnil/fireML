import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Path to your NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_202010.nc'

# Open the NetCDF file
dataset = nc.Dataset(file_path, mode='r')

# Print all variables in the file to explore
# print("Variables in the file:", dataset.variables.keys())
'''
Variables in the file: 
dict_keys(['west_east', 'south_north', 'XLAT_M', 'XLONG_M', 
            'Merged_BurnDate', 'MTBS_BurnDate', 'NIFC_BurnDate', 'MCD_BurnDate', 
            'MTBS_BurnFraction', 'NIFC_BurnFraction'])
'''

# Extracting the necessary variables
burn_area = dataset.variables['MTBS_BurnFraction'][:]  # Burn area data
lat = dataset.variables['XLAT_M'][:]  # Latitude
lon = dataset.variables['XLONG_M'][:]  # Longitude

# Check if lat, lon, and burn_area have the same dimensions
print("lat shape:", lat.shape)
print("lon shape:", lon.shape)
print("burn_area shape:", burn_area.shape)

# Handle NaN or non-finite values in the data (lat, lon, burn_area)
# Replace NaN or infinite values with zeros or mask them out
burn_area = np.ma.masked_invalid(burn_area)
lat = np.ma.masked_invalid(lat)
lon = np.ma.masked_invalid(lon)

# Ensure all arrays have valid finite values
valid_mask = (~burn_area.mask) & (~lat.mask) & (~lon.mask)

# Use the valid_mask to filter out invalid data points
filtered_burn_area = burn_area[valid_mask]
filtered_lat = lat[valid_mask]
filtered_lon = lon[valid_mask]

# Plotting the filtered BurnArea data
plt.figure(figsize=(10, 8))
plt.scatter(filtered_lon, filtered_lat, c=filtered_burn_area, cmap='inferno', s=1)
plt.colorbar(label='Burn Fraction')  # Add colorbar for burn fraction
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Filtered Burn Area Visualization for September 2020')
plt.show()

# Close the NetCDF file
dataset.close()
