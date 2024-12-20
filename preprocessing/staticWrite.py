import netCDF4 as nc
import numpy as np

# Path for the existing NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/data/Static_data_all.nc'

# Open the existing NetCDF file in 'r+' mode to allow reading and writing
output_file = nc.Dataset(file_path, 'r+', format='NETCDF4')

# Define the dimensions (use existing dimensions if they already exist)
if 'ncoords' not in output_file.dimensions:
    ncoords = output_file.createDimension('ncoords', None)  # Create the dimension only if it doesn't exist

# Create the new variables
if 'firstBurn' not in output_file.variables:
    firstBurn = output_file.createVariable('firstBurn', np.float32, ('ncoords',))

if 'finalBurn' not in output_file.variables:
    finalBurn = output_file.createVariable('finalBurn', np.float32, ('ncoords',))

# (Values will be assigned later)

# Close the file after making modifications
output_file.close()

print(f"NetCDF file updated with firstBurn and finalBurn variables at {file_path}")
