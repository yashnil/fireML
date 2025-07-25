#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def gather_features(ds, target_var="DOD"):
    """
    Gathers all variables from `ds` except:
     - the target var (DOD),
     - lat/lon/pixel/year or dimension/index vars,
     - anything that obviously shouldn't be a predictor (e.g. 'ncoords_vector').

    If a variable has shape (15, pixel), we average over the year dimension => shape (pixel,).
    If it has shape (pixel,), we take it as is.
    Returns a dict of {var_name: 1D array} with shape (pixel,) for each.
    """
    exclude_vars = {
        target_var.lower(),  # 'dod'
        'lat', 'lon', 'latitude', 'longitude',
        'pixel', 'year', 'ncoords_vector', 'nyears_vector'  # or any other index coords
    }
    all_features = {}

    for var_name in ds.data_vars:
        if var_name.lower() in exclude_vars:
            continue  # skip

        # shape info
        da = ds[var_name]
        dims = da.dims
        # If dimension is (year, pixel=...) => average over 'year'
        if 'year' in dims:
            # e.g. shape (15, pixel)
            # average over year => shape (pixel,)
            arr_2d = da.values  # (year, pixel)
            arr_1d = np.nanmean(arr_2d, axis=0)  # shape (pixel,)
        else:
            # assume shape (pixel,)
            arr_1d = da.values

        # Make sure it is 1D => shape(pixel,)
        if arr_1d.ndim != 1:
            # skip if shape is not (pixel,)
            continue

        # store in all_features
        all_features[var_name] = arr_1d

    return all_features


def train_model_with_all_features(
    dataset_path="/Users/yashnilmohanty/Desktop/final_dataset2.nc",
    burn_var="burn_fraction",
    target_var="DOD",  # We predict DOD
    burn_threshold=0.7
):
    """
    1) Loads final_dataset2.nc
    2) Gathers all variables except 'DOD' (target) + lat/lon/pixel dimension => X
    3) Averages each var across year if shape=(15, pixel), or takes as is if shape=(pixel,)
    4) 'DOD' is also averaged across year => y (predictand)
    5) Defines 'burn_sum' from 'burn_fraction' => sum across year => unburned vs. burned
    6) Trains on unburned => splits train/test
    7) Tests on unburned holdout + all burned
    """

    print(f"Loading dataset: {dataset_path}")
    ds = xr.open_dataset(dataset_path)

    # 1) gather all features except DOD
    all_feats = gather_features(ds, target_var=target_var)

    # Convert this dict -> DataFrame => shape (pixel, n_features)
    feat_names = sorted(all_feats.keys())
    arr_list   = []
    for fname in feat_names:
        arr_list.append(all_feats[fname])
    X_all = np.column_stack(arr_list)  # shape (pixel, n_features)

    # 2) define target (DOD) => average across year => shape(pixel,)
    dod_2d = ds[target_var].values  # shape (15, pixel)
    y_all  = np.nanmean(dod_2d, axis=0)  # shape(pixel,)

    # 3) define unburned vs burned from 'burn_fraction'
    burn_2d = ds[burn_var].values  # shape(15, pixel)
    burn_sum = np.nansum(burn_2d, axis=0)  # shape(pixel,)
    unburned_mask = burn_sum < burn_threshold
    burned_mask   = ~unburned_mask

    # 4) remove any NaN from X_all or y_all
    # shape must match => #pixel rows
    valid_mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)
    # combine with unburned/burned
    unburned_mask = unburned_mask & valid_mask
    burned_mask   = burned_mask & valid_mask

    # Some stats
    n_total    = X_all.shape[0]
    n_unburned = np.sum(unburned_mask)
    n_burned   = np.sum(burned_mask)
    print(f"Total pixels: {n_total}")
    print(f"  valid_mask => {np.sum(valid_mask)} used")
    print(f"  unburned => {n_unburned}")
    print(f"  burned   => {n_burned}")

    if n_unburned == 0:
        print("No unburned pixels => cannot train model.")
        return

    # 5) Build X_unburned, y_unburned
    X_unburned = X_all[unburned_mask]
    y_unburned = y_all[unburned_mask]

    # Build X_burned, y_burned
    X_burned = X_all[burned_mask]
    y_burned = y_all[burned_mask]

    # 6) train/test split among unburned
    from sklearn.model_selection import train_test_split
    X_train, X_test_unburn, y_train, y_test_unburn = train_test_split(
        X_unburned, y_unburned, test_size=0.3, random_state=42
    )
    print(f"  unburned train: {len(X_train)}, unburned test: {len(X_test_unburn)}")

    # 7) Model => random forest regressor
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # evaluate on unburned test
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred_unburn = model.predict(X_test_unburn)
    mse_unburn = mean_squared_error(y_test_unburn, y_pred_unburn)
    r2_unburn  = r2_score(y_test_unburn, y_pred_unburn)
    print(f"\nUnburned Test => MSE: {mse_unburn:.4f}, R^2: {r2_unburn:.4f}")

    # evaluate on ALL burned
    if len(X_burned) > 0:
        y_pred_burn = model.predict(X_burned)
        mse_burn = mean_squared_error(y_burned, y_pred_burn)
        r2_burn  = r2_score(y_burned, y_pred_burn)
        print(f"Burned => MSE: {mse_burn:.4f}, R^2: {r2_burn:.4f}, #samples: {len(X_burned)}")

    # Feature importances
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for fname, imp in zip(feat_names, importances):
        print(f"  {fname}: {imp:.4f}")

if __name__ == "__main__":
    train_model_with_all_features(
        dataset_path="/Users/yashnilmohanty/Desktop/final_dataset2.nc",
        burn_var="burn_fraction",
        target_var="DOD",
        burn_threshold=0.7
    )
