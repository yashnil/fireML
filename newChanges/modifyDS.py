import netCDF4 as nc
import numpy as np
import os

# ---------------------------------------------------------------------
# 1. Identify which AORC variables must be shifted back by one year
#    (the "Fall" and "Summer" variables in final_dataset2).
# ---------------------------------------------------------------------
SHIFT_BACK_VARIABLES = [
    # Summer
    "aorcSummerTemperature",
    "aorcSummerRain",         # renamed to aorcSummerPrecipitation
    "aorcSummerHumidity",
    "aorcSummerShortwave",
    "aorcSummerLongwave",
    # Fall
    "aorcFallTemperature",
    "aorcFallRain",           # renamed to aorcFallPrecipitation
    "aorcFallHumidity",
    "aorcFallShortwave",
    "aorcFallLongwave"
]

# ---------------------------------------------------------------------
# 2. Map final_dataset2 varname -> (season, raw AORC var).
# ---------------------------------------------------------------------
AORC_VAR_MAP = {
    "aorcFallTemperature":    ("Fall",   "T2D"),
    "aorcFallRain":           ("Fall",   "RAINRATE"),
    "aorcFallHumidity":       ("Fall",   "specifc_humidity"),
    "aorcFallShortwave":      ("Fall",   "SWDOWN"),
    "aorcFallLongwave":       ("Fall",   "LWDOWN"),

    "aorcSummerTemperature":  ("Summer", "T2D"),
    "aorcSummerRain":         ("Summer", "RAINRATE"),
    "aorcSummerHumidity":     ("Summer", "specifc_humidity"),
    "aorcSummerShortwave":    ("Summer", "SWDOWN"),
    "aorcSummerLongwave":     ("Summer", "LWDOWN")
}

# ---------------------------------------------------------------------
# 3. Helper to compute seasonal means from the *previous* year's file.
# ---------------------------------------------------------------------
def compute_aorc_shifted_means(aorc_season, aorc_varname, start_year=2004, end_year=2018,
                               pixel_indices=None,
                               base_dir="/Users/yashnilmohanty/Desktop/data/AORC_Data_Northen_CA"):
    """
    For each year Y in [start_year..end_year], open the netCDF from (Y-1),
    read 'aorc_varname', average over the time dimension, and subselect
    'pixel_indices'. Return array shape = (15, len(pixel_indices)).
    """
    years = np.arange(start_year, end_year + 1)
    out_data = np.full((len(years), len(pixel_indices)), np.nan, dtype=np.float32)

    for i, yr in enumerate(years):
        prev_year = yr - 1
        file_name = f"AORC_Data_Northen_CA_{aorc_season}_CY{prev_year}.nc"
        file_path = os.path.join(base_dir, file_name)

        with nc.Dataset(file_path, 'r') as ds_in:
            # shape: (Time, 95324)
            raw_data = ds_in.variables[aorc_varname][:]
            mean_over_time = np.mean(raw_data, axis=0)  # shape: (95324,)

            # final_dataset2's pixel dimension is 0..95323, so we can directly index
            subsel = mean_over_time[pixel_indices]
            out_data[i, :] = subsel.astype(np.float32)

    return out_data

# ---------------------------------------------------------------------
# 4. Utility to rename "Rain" -> "Precipitation" in the final variable name.
# ---------------------------------------------------------------------
def rename_rain_to_precip(var_name):
    if "Rain" in var_name:
        return var_name.replace("Rain", "Precipitation")
    return var_name

# ---------------------------------------------------------------------
# 5. Main code to read final_dataset2.nc and produce final_dataset3.nc
# ---------------------------------------------------------------------
input_path  = "/Users/yashnilmohanty/Desktop/final_dataset2.nc"
output_path = "/Users/yashnilmohanty/Desktop/final_dataset3.nc"

with nc.Dataset(input_path, 'r') as ds_in, nc.Dataset(output_path, 'w') as ds_out:
    
    # Copy global attributes (optional but often helpful)
    for attr_name in ds_in.ncattrs():
        setattr(ds_out, attr_name, getattr(ds_in, attr_name))

    # Copy dimensions
    for dname, dim in ds_in.dimensions.items():
        ds_out.createDimension(dname, len(dim) if not dim.isunlimited() else None)

    # Read pixel indices (0..95323) from final_dataset2
    pixel_var_in = ds_in.variables['pixel']
    pixel_indices = pixel_var_in[:].astype(int)  # ensure integer

    # Create the 'pixel' variable in ds_out. 
    # We handle _FillValue here at creation time if present.
    fill_val = pixel_var_in.getncattr('_FillValue') if '_FillValue' in pixel_var_in.ncattrs() else None
    pixel_var_out = ds_out.createVariable('pixel', 'i4', ('pixel',), fill_value=fill_val)

    # Copy over attributes except _FillValue
    for attr in pixel_var_in.ncattrs():
        if attr == '_FillValue':
            continue
        setattr(pixel_var_out, attr, pixel_var_in.getncattr(attr))
    pixel_var_out[:] = pixel_indices

    # Similarly handle 'year' or 'nyears_vector'
    # (Use whichever dimension/variable name your dataset actually has.)
    if 'year' in ds_in.variables:
        year_in = ds_in.variables['year']
        fv = year_in.getncattr('_FillValue') if '_FillValue' in year_in.ncattrs() else None
        year_out = ds_out.createVariable('year', year_in.dtype, ('year',), fill_value=fv)
        for attr in year_in.ncattrs():
            if attr == '_FillValue':
                continue
            setattr(year_out, attr, year_in.getncattr(attr))
        year_out[:] = year_in[:]

    if 'nyears_vector' in ds_in.variables:
        nyears_in = ds_in.variables['nyears_vector']
        fv = nyears_in.getncattr('_FillValue') if '_FillValue' in nyears_in.ncattrs() else None
        nyears_out = ds_out.createVariable('nyears_vector', nyears_in.dtype, ('year',), fill_value=fv)
        for attr in nyears_in.ncattrs():
            if attr == '_FillValue':
                continue
            setattr(nyears_out, attr, nyears_in.getncattr(attr))
        nyears_out[:] = nyears_in[:]

    # Now copy (or fix) each remaining variable
    for var_name, var_in in ds_in.variables.items():
        # We've already handled 'pixel', 'year', 'nyears_vector'
        if var_name in ('pixel','year','nyears_vector'):
            continue

        # Figure out if we must shift the data by 1 year (Fall/Summer)
        if var_name in SHIFT_BACK_VARIABLES:
            season, raw_aorc_var = AORC_VAR_MAP[var_name]
            # Recompute the data from (year-1)
            data_out = compute_aorc_shifted_means(
                aorc_season=season,
                aorc_varname=raw_aorc_var,
                start_year=2004,
                end_year=2018,
                pixel_indices=pixel_indices
            )
            new_var_name = rename_rain_to_precip(var_name)

            # Create variable in ds_out
            fv = var_in.getncattr('_FillValue') if '_FillValue' in var_in.ncattrs() else None
            out_var = ds_out.createVariable(new_var_name,
                                            np.float32,
                                            var_in.dimensions,
                                            fill_value=fv)

            # Copy over attributes except _FillValue
            for attr in var_in.ncattrs():
                if attr == '_FillValue':
                    continue
                setattr(out_var, attr, var_in.getncattr(attr))

            # Assign data
            out_var[:] = data_out

        else:
            # Just copy the data over (with possible rename from 'Rain' to 'Precipitation')
            new_var_name = rename_rain_to_precip(var_name)
            fv = var_in.getncattr('_FillValue') if '_FillValue' in var_in.ncattrs() else None
            out_var = ds_out.createVariable(new_var_name,
                                            var_in.dtype,
                                            var_in.dimensions,
                                            fill_value=fv)
            for attr in var_in.ncattrs():
                if attr == '_FillValue':
                    continue
                setattr(out_var, attr, var_in.getncattr(attr))

            out_var[:] = var_in[:]

print(f"Successfully created {output_path} with corrected Summer/Fall data and renamed variables.")
