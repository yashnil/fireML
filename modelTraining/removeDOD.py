import xarray as xr
import numpy as np

# Load the NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/adjusted_combined_data.nc'  # Update with your actual file path
data = xr.open_dataset(file_path)

# Step 1: Flatten Geo2D data
if 'lat' in data.dims and 'lon' in data.dims:
    lat, lon = np.meshgrid(data['lat'].values, data['lon'].values, indexing='ij')
    flat_lat = lat.flatten()
    flat_lon = lon.flatten()
    
    # Create a new spatial dimension 'npixels'
    n_pixels = len(flat_lat)
    data = data.assign_coords(npixels=('npixels', np.arange(n_pixels)))
    data['lat'] = (('npixels',), flat_lat)
    data['lon'] = (('npixels',), flat_lon)
    
    # Flatten all Geo2D variables
    for feature in data.variables:
        if 'lat' in data[feature].dims and 'lon' in data[feature].dims:
            flat_data = data[feature].values.reshape(-1)  # Flatten the data
            data[feature] = xr.DataArray(
                flat_data,
                dims=('npixels',),
                coords={'npixels': data['npixels']}
            )
        elif 'lat' in data[feature].dims or 'lon' in data[feature].dims:
            raise ValueError(f"Unexpected dimensions in {feature}. Check the dataset.")

# Step 2: Ensure regular 2D data uses the same 'npixels' dimension
# Replace 'ncoords' with 'npixels' for variables with the 'ncoords' dimension
if 'ncoords' in data.dims:
    for feature in data.variables:
        if 'ncoords' in data[feature].dims:
            data[feature] = data[feature].rename({'ncoords': 'npixels'})

# Step 3: Extract DOD and identify NaN columns
dod = data['DOD']  # Extract the DOD feature

# Ensure DOD uses the 'npixels' dimension
if 'ncoords' in dod.dims:
    dod = dod.rename({'ncoords': 'npixels'})

# Identify columns with NaN values and create a mask
nan_columns = np.any(np.isnan(dod), axis=0)  # Identify columns with NaN values
columns_to_keep = ~nan_columns  # Mask for columns to keep (no NaN values in DOD)

# Step 4: Filter all features based on the columns to keep
processed_data = {}
for feature in data.variables:
    feature_dims = data[feature].dims
    feature_data = data[feature]

    if 'npixels' in feature_dims:
        # Filter features with spatial coordinates
        processed_data[feature] = xr.DataArray(
            feature_data.isel(npixels=columns_to_keep).data,
            dims=feature_dims,
            coords={dim: feature_data[dim] for dim in feature_dims if dim != 'npixels'}
        )
    else:
        # Retain features without spatial dimensions
        processed_data[feature] = feature_data

# Step 5: Save the preprocessed data to a new NetCDF file
preprocessed_file_path = '/Users/yashnilmohanty/Desktop/preprocessed_combined_data.nc'  # Update with your output path
processed_dataset = xr.Dataset(processed_data)
processed_dataset.to_netcdf(preprocessed_file_path)

# Print summary
num_initial_coords = dod.sizes['npixels']
num_remaining_coords = np.sum(columns_to_keep)
num_dropped_coords = num_initial_coords - num_remaining_coords

print(f"Initial Coordinates: {num_initial_coords}")
print(f"Remaining Coordinates: {num_remaining_coords}")
print(f"Dropped Coordinates: {num_dropped_coords}")
