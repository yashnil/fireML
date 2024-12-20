import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

from preprocessing.getLatLong import coords

# Load your external coordinates (95324 × 2 array with lat/lon)
coordinates = coords
# Define the years and months to process
years = range(2004, 2019)  # 2004 to 2018 inclusive
months = range(1, 13)  # January to December

# Initialize an empty DataFrame to store results
final_df = pd.DataFrame(index=years, columns=range(95324))

# Loop over each year
for year in years:
    yearly_sum = np.zeros((95324,))  # To store cumulative burn fraction for the year

    # Create a separate log file for each year
    log_file_path = f"/Users/yashnilmohanty/Desktop/output_log_{year}.txt"
    with open(log_file_path, "w") as log_file:
        # Loop over each month
        for month in months:
            file_path = f'/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc'

            # Open the NetCDF file
            dataset = nc.Dataset(file_path, mode='r')

            # Extract the necessary variables
            burn_area = dataset.variables['MTBS_BurnFraction'][:]  # Burn fraction data
            lat = dataset.variables['XLAT_M'][:]
            lon = dataset.variables['XLONG_M'][:]

            # Flatten the lat/lon and burn_area grids
            flat_lat = lat.flatten()
            flat_lon = lon.flatten()
            flat_burn_area = burn_area.flatten()

            # Remove invalid (NaN/inf) entries
            valid_mask = np.isfinite(flat_lat) & np.isfinite(flat_lon) & np.isfinite(flat_burn_area)
            valid_lat = flat_lat[valid_mask]
            valid_lon = flat_lon[valid_mask]
            valid_burn_area = flat_burn_area[valid_mask]

            # Loop over each coordinate to calculate burn fraction
            for i, (coord_lat, coord_lon) in enumerate(coordinates):
                # Define the bounding box
                lat_min, lat_max = coord_lat - 0.005, coord_lat + 0.005  # ±0.005 degrees ≈ ±500 m
                lon_min, lon_max = coord_lon - 0.005, coord_lon + 0.005

                # Find points within the bounding box
                in_box = (valid_lat >= lat_min) & (valid_lat <= lat_max) & \
                         (valid_lon >= lon_min) & (valid_lon <= lon_max)

                # Filter points within the bounding box
                box_burn_area = valid_burn_area[in_box]

                # Log the details to the file
                #if len(box_burn_area) > 0:
                #    log_file.write(
                #        f"Year: {year}, Month: {month}, Coordinate {i}: {len(box_burn_area)} points in bounding box\n"
                #    )

                if len(box_burn_area) >= 12:
                    log_file.write(
                        f"Hard stop at Coordinate {i}, 12 points found in bounding box\n"
                    )
                    break  # Exit the loop if exactly 12 points are found

                if not len(box_burn_area):
                    mean_fraction = np.nan
                else:
                    mean_fraction = np.mean(box_burn_area)

                if not np.isnan(mean_fraction):
                    yearly_sum[i] += mean_fraction

            # Close the NetCDF file
            dataset.close()

        # Log completion message
        # log_file.write("Processing complete for year {year}.\n")

    # Store the cumulative sum for the year in the final dataframe
    final_df.loc[year] = yearly_sum

    # Notify the terminal when year is fully processed. 
    print("Finished Processing Data for year " + str(year))

# Unpack the coordinates into latitude and longitude arrays
latitudes, longitudes = map(list,zip(*coordinates))  # Extract lat and lon as separate lists

# Convert the final DataFrame to an xarray DataArray
xr_data = xr.DataArray(
    data=final_df.values,
    dims=['year', 'location'],
    coords={
        'year': final_df.index,          # Year index
        'location': range(95324),       # Location indices
        'latitude': ('location', latitudes),  # Latitude for each location
        'longitude': ('location', longitudes)  # Longitude for each location
    }
)

# Create a dataset
ds = xr.Dataset({'burn_fraction': xr_data})

# Save the dataset to a NetCDF file
output_path = '/Users/yashnilmohanty/Desktop/fire.nc'
ds.to_netcdf(output_path)

print(f'NetCDF file successfully written to {output_path}')
