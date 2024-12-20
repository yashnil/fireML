
import netCDF4 as nc
import numpy as np
import scipy.io as sio  # For loading .mat files

# Create output NetCDF file
output_file = nc.Dataset('/Users/yashnilmohanty/Desktop/finalDF.nc', 'w', format='NETCDF4')

# Function to read and compute means for a given variable and season
def process_variable(season, variable_name, start_year, end_year):
    variable_means = []
    for i in range(start_year, end_year + 1):
        file_path = f'/Users/yashnilmohanty/Desktop/data/AORC_Data_Northen_CA/AORC_Data_Northen_CA_{season}_CY{i}.nc'
        dataset = nc.Dataset(file_path, mode='r')
        
        variable_data = dataset.variables[variable_name][:]  # Shape (time, ncoords)
        variable_mean = np.mean(variable_data, axis=0)  # Average over the time dimension (axis 0)
        variable_means.append(variable_mean)

    return np.array(variable_means)  # Convert list of means to a numpy array

# Function to process SWE data for winter (days 32:124) and spring (day 182/183)
def process_swe_data(start_year, end_year):
    swe_winter_means = []
    swe_spring_values = []
    
    for i in range(start_year, end_year + 1):
        file_path = f'/Users/yashnilmohanty/Desktop/data/SNODAS_SWE_by_WY/SNODAS_SWE_{i}.nc'
        dataset = nc.Dataset(file_path, mode='r')
        
        swe_data = dataset.variables['SWE'][:]  # Shape (time, ncoords), 365 or 366 days
        
        # Winter: average over days 32 to 124
        swe_winter_mean = np.mean(swe_data[32:125, :], axis=0)  # 125 is exclusive, so includes up to 124
        swe_winter_means.append(swe_winter_mean)
        
        # Spring: day 182 for normal year, day 183 for leap year
        if i % 4 == 0 and (i % 100 != 0 or i % 400 == 0):  # Leap year condition
            swe_spring_value = swe_data[183, :]
        else:
            swe_spring_value = swe_data[182, :]
        swe_spring_values.append(swe_spring_value)
    
    return np.array(swe_winter_means), np.array(swe_spring_values)

# Function to process DOD data from .mat files
def process_dod_data(start_year, end_year):
    dod_values = []
    
    for i in range(start_year, end_year + 1):
        file_path = f'/Users/yashnilmohanty/Desktop/data/ORNL_DOD_north_CA_by_CY/ORNL_DOD_nCA_CY{i}.mat'
        data = sio.loadmat(file_path)
        
        # Extract the 'DOD_nCA' variable, assuming it's in the same format as described (95324 coordinates)
        dod_data = data['DOD_nCA'][:, 2]  # Column 2 contains the DOD values
        dod_values.append(dod_data)

    return np.array(dod_values)  # Convert list of DOD data to a numpy array

# Create dimensions for years and coordinates (only once for all variables)
nyears = output_file.createDimension('nyears', 15)  # 15 years from 2004 to 2018
ncoords = None

# Create the 1D nyears vector
years = np.arange(2004, 2019)  # Years 2004 to 2018
nyears_var = output_file.createVariable('nyears_vector', np.int32, ('nyears',))
nyears_var[:] = years

# Assuming coordinates can be taken from the first AORC dataset or SWE data
first_swe_file = '/Users/yashnilmohanty/Desktop/data/SNODAS_SWE_by_WY/SNODAS_SWE_2004.nc'
first_swe_dataset = nc.Dataset(first_swe_file, mode='r')
coords = first_swe_dataset.variables['total_length'][:]  # Assuming 'total_length' is the coordinate array

# Create ncoords dimension based on the length of the coords array
if ncoords is None:
    ncoords = output_file.createDimension('ncoords', len(coords))  # Assuming 95324 coordinates

# Create the 1D ncoords vector
ncoords_var = output_file.createVariable('ncoords_vector', np.float32, ('ncoords',))
ncoords_var[:] = coords

# Process variables for each season
seasons = ['Fall', 'Winter', 'Spring', 'Summer']
variables = ['T2D', 'RAINRATE', 'specifc_humidity', 'SWDOWN', 'LWDOWN']
variableNames = ['Temperature', 'Rain', 'Humidity', 'Shortwave', 'Longwave']

for season in seasons:
    for variable in variables:
        var_means_array = process_variable(season, variable, 2004, 2018)

        # Create ncoords dimension if it hasn't been created yet
        if ncoords is None:
            ncoords = output_file.createDimension('ncoords', len(var_means_array[0]))  # 95324 coordinates

        # Create variable in NetCDF file (e.g., aorcFallTemperature, aorcWinterRain, etc.)
        variable_name = f'aorc{season}{variableNames[variables.index(variable)]}'
        output_variable = output_file.createVariable(variable_name, np.float32, ('nyears', 'ncoords'))
        output_variable[:, :] = var_means_array

# Process SWE data for winter and spring
swe_winter_means, swe_spring_values = process_swe_data(2004, 2018)

# Create SWE variables in the NetCDF file
swe_winter_var = output_file.createVariable('sweWinter', np.float32, ('nyears', 'ncoords'))
swe_winter_var[:, :] = swe_winter_means

swe_spring_var = output_file.createVariable('sweSpring', np.float32, ('nyears', 'ncoords'))
swe_spring_var[:, :] = swe_spring_values

# Process DOD data and add to the NetCDF file
dod_data = process_dod_data(2004, 2018)

# Create DOD variable in the NetCDF file
dod_var = output_file.createVariable('DOD', np.float32, ('nyears', 'ncoords'))
dod_var[:, :] = dod_data

# Close the NetCDF file
output_file.close()

print(f"Data successfully written to finalDF.nc")
