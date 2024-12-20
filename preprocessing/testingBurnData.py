
import xarray as xr

# Open the netCDF file
data = xr.open_dataset("/Users/yashnilmohanty/Desktop/fire.nc")

# Extract the burn_fraction variable
burn_fraction = data["burn_fraction"]

# Find coordinates where burn_fraction is 0 for all years
zero_burn_coordinates = (burn_fraction == 0).all(dim="year")

# Count the number of such coordinates
num_zero_burn_coordinates = zero_burn_coordinates.sum().item()

print(f"Number of coordinates with 0 burn fraction for all years: {num_zero_burn_coordinates}")
