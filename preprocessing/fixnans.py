
import netCDF4 as nc
import numpy as np

# Path to the existing NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/finalDF.nc'

# Open the NetCDF file in read-write mode
with nc.Dataset(file_path, mode='r+') as dataset:
    # Access the sweWinter variable
    swe_winter = dataset.variables['sweWinter'][:]  # Shape: (nyears, ncoords)
    swe_spring = dataset.variables['sweSpring'][:]  # Shape: (nyears, ncoords)
    
    # Fix NaN values in sweWinter (e.g., replace with the mean across all valid years)
    print("Fixing NaN values in sweWinter...")
    swe_winter_mean = np.nanmean(swe_winter, axis=0)  # Compute mean for each coordinate
    nan_mask_winter = np.isnan(swe_winter)  # Identify NaNs
    swe_winter[nan_mask_winter] = np.take(swe_winter_mean, np.where(nan_mask_winter)[1])  # Replace NaNs with mean
    
    # Fix NaN values in sweSpring (similar approach)
    print("Fixing NaN values in sweSpring...")
    swe_spring_mean = np.nanmean(swe_spring, axis=0)  # Compute mean for each coordinate
    nan_mask_spring = np.isnan(swe_spring)  # Identify NaNs
    swe_spring[nan_mask_spring] = np.take(swe_spring_mean, np.where(nan_mask_spring)[1])  # Replace NaNs with mean
    
    # Write the fixed data back to the NetCDF file
    dataset.variables['sweWinter'][:] = swe_winter
    dataset.variables['sweSpring'][:] = swe_spring

print("NaN values have been fixed in sweWinter and sweSpring.")
