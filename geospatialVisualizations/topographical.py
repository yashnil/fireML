import netCDF4 as nc
import matplotlib.pyplot as plt

# Replace 'your_file.nc' with the path to your .nc file
file_path = '/Users/yashnilmohanty/Desktop/data/Static_data_all.nc'

# Open the NetCDF file
dataset = nc.Dataset(file_path, mode='r')

# To see the structure of the dataset
print(dataset.variables.keys())

# Example: Plotting elevation data
elevation = dataset.variables['Elevation'][:]
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

print("Latitude dimension:", lat.shape)
print("Longitude dimension:", lon.shape)

plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=elevation, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Elevation Map')
plt.show()
