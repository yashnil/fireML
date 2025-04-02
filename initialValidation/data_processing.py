# data_processing.py

import xarray as xr
import numpy as np

def load_data(burn_file, dod_file, burn_varname='burn_fraction', dod_varname='DoD'):
    """
    Loads burn fraction and DoD from netCDF files.
    'fire_modified.nc' -> shape (15, 95324)
    'finalDF.nc' -> shape (15, 95324)
    """
    ds_burn = xr.open_dataset(burn_file)
    burn_array = ds_burn[burn_varname].values  # shape: (15, 95324)

    ds_dod = xr.open_dataset(dod_file)
    dod_array = ds_dod[dod_varname].values     # shape: (15, 95324)

    return burn_array, dod_array

def remove_pixels_if_any_invalid_dod(burn_array, dod_array):
    """
    Removes entire pixels that have ANY invalid DoD (NaN, <=0) in any of the 15 years.
    Returns:
        burn_array_sub, dod_array_sub (15, num_valid_pixels),
        valid_pixels (the pixel IDs that remain).
    """
    n_years, n_pixels = dod_array.shape
    valid_pixels = []

    for pix in range(n_pixels):
        pixel_dod = dod_array[:, pix]  # shape (15,)
        # invalid if any year is NaN
        if not np.isnan(pixel_dod).any():
            valid_pixels.append(pix)

    valid_pixels = np.array(valid_pixels)
    burn_array_sub = burn_array[:, valid_pixels]
    dod_array_sub  = dod_array[:, valid_pixels]

    return burn_array_sub, dod_array_sub, valid_pixels

def classify_burned_unburned(burn_array, threshold=0.05):
    """
    Sum the 15-year burn fraction per pixel; classify as unburned (<threshold) or burned (>= threshold).
    Returns unburned_mask, burned_mask.
    """
    cumulative_burn = np.sum(burn_array, axis=0)  # shape: (num_valid_pixels,)
    unburned_mask = (cumulative_burn < threshold)
    burned_mask   = ~unburned_mask
    return unburned_mask, burned_mask

def split_unburned_pixels(unburned_mask, train_ratio=0.8, seed=42):
    """
    Splits unburned pixels into train vs test by train_ratio.
    Returns train_inds, test_inds (indices within the final subset).
    """
    unburned_indices = np.where(unburned_mask)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(unburned_indices)
    train_size = int(train_ratio * len(unburned_indices))
    train_inds = unburned_indices[:train_size]
    test_inds  = unburned_indices[train_size:]
    return train_inds, test_inds
