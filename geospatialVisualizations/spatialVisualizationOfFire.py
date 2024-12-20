import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Open the netCDF file
data = xr.open_dataset("/Users/yashnilmohanty/Desktop/fire.nc")

# Extract relevant variables
burn_fraction = data["burn_fraction"]
latitude = data["latitude"]
longitude = data["longitude"]
year = data["year"]

# Calculate the mean burn fraction over all years for each coordinate point
mean_burn_fraction = burn_fraction.mean(dim="year")

# Ensure "location" dimension exists to map to latitude and longitude
if "location" not in mean_burn_fraction.dims:
    raise ValueError("Location dimension is missing in the dataset")

# Extract latitude, longitude, and burn fraction as flat arrays
latitudes = latitude.values
longitudes = longitude.values
mean_burn_fraction_values = mean_burn_fraction.values

# Create a scatter plot to visualize spatial data
plt.figure(figsize=(10, 8))

# Scatter plot with lat/lon and color representing burn fraction
sc = plt.scatter(longitudes, latitudes, c=mean_burn_fraction_values, cmap="hot", s=1, edgecolor='none')
plt.colorbar(sc, label="Mean Burn Fraction")

# Add labels and a title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Mean Burn Fraction")

# Show the plot
plt.show()