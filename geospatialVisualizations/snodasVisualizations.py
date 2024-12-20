import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Open the NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/data/SNODAS_SWE_by_WY/SNODAS_SWE_2013.nc'
dataset = nc.Dataset(file_path, mode='r')

for variable in dataset.variables:
    print(variable)

length = dataset.variables['total_length'][:] # Shape (95324,)
time = dataset.variables['Time'][:]  # Shape (366,)
swe = dataset.variables['SWE'][:]    # Shape (366, 95324)

# Replace NaNs with the average value of non-NaN values in the SWE dataset
mean_swe = np.nanmean(swe, axis=1)
filled_swe = np.where(np.isnan(swe), mean_swe[:, None], swe)
mean_swe = np.nanmean(swe, axis=1)
# Compute the average SWE per day after interpolation
average_swe_per_day_interpolated = np.mean(filled_swe, axis=1)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(time, average_swe_per_day_interpolated, label='Average SWE (NaNs Interpolated)')
plt.xlabel('Time (days)')
plt.ylabel('SWE (mm)')
plt.title('Average Snow Water Equivalent (SWE) Over Time (NaNs Interpolated)')
plt.legend()
plt.grid(True)
plt.show()

'''
# Compute the average SWE per day while ignoring NaN values
average_swe_per_day = np.nanmean(swe, axis=1)

# Plot the average SWE over time
plt.figure(figsize=(10, 6))
plt.plot(time, average_swe_per_day, label='Average SWE')
plt.xlabel('Time (days)')
plt.ylabel('SWE (mm)')
plt.title('Average Snow Water Equivalent (SWE) Over Time (NaNs Ignored)')
plt.legend()
plt.grid(True)
plt.show()
'''