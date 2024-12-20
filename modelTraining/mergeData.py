import xarray as xr

# Step 1: Load the datasets
static_data = xr.open_dataset("/Users/yashnilmohanty/Desktop/data/Static_data_all.nc")
meteor_data = xr.open_dataset("/Users/yashnilmohanty/Desktop/finalDF.nc")

# Step 2: Inspect dimensions
print("Static Data Dimensions:", static_data.dims)
print("Meteorological Data Dimensions:", meteor_data.dims)

# Step 3: Drop 'ncoords' dimension from static_data to resolve conflicts
static_data = static_data.drop_dims('ncoords', errors='ignore')

# Step 4: Prepare static data for broadcasting
# Set 'nyears_vector' as a coordinate
meteor_data = meteor_data.set_coords('nyears_vector')

# Extract the years as a numpy array
years = meteor_data.coords['nyears_vector']  # Extract the year coordinate
static_data_expanded = static_data.expand_dims(year=years.values)

# Now, static_data_expanded has dimensions (npixels, year) matching meteor_data

# Step 5: Merge the static and meteorological datasets
combined_data = xr.merge([static_data_expanded, meteor_data])

# Step 6: Verify the combined dataset
print("Combined Data Dimensions:", combined_data.dims)
print("Combined Data Variables:", combined_data.data_vars)

# Optional: Save the combined dataset
combined_data.to_netcdf("/Users/yashnilmohanty/Desktop/combined_data.nc")
print("Combined dataset saved as combined_data.nc")
