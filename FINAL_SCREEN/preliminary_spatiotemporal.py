#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def gather_spatiotemporal_features(ds, target_var="DOD"):
    """
    Gathers all variables from `ds` as spatiotemporal features.
    We skip:
      - the target var (DOD),
      - lat/lon/pixel/year dimension variables,
      - known index variables,
      - anything with shape not matching (year,pixel) or (pixel,).

    For shape (year,pixel), we keep as is.
    For shape (pixel,), we replicate across year dimension -> shape (year,pixel).
    Returns {var_name: 2D array (year, pixel)}.
    """
    exclude_vars = {
        target_var.lower(),
        'lat', 'lon', 'latitude', 'longitude',
        'pixel', 'year', 'ncoords_vector', 'nyears_vector'
    }
    all_features = {}

    # The dataset should have dims: year=15, pixel=whatever
    n_years = ds.dims['year']

    for var_name in ds.data_vars:
        if var_name.lower() in exclude_vars:
            continue

        da = ds[var_name]
        dims = da.dims

        # shape (year, pixel) => keep as is
        if set(dims) == {'year', 'pixel'}:
            arr_2d = da.values  # shape (year, pixel)
            all_features[var_name] = arr_2d

        # shape (pixel,) => replicate across year
        elif set(dims) == {'pixel'}:
            arr_1d = da.values  # shape (pixel,)
            arr_2d = np.tile(arr_1d, (n_years, 1))  # shape (year, pixel)
            all_features[var_name] = arr_2d

        else:
            # skip variables that don't match these shapes
            continue

    return all_features

def spatiotemporal_model_with_pixelSum(
    dataset_path="/Users/yashnilmohanty/Desktop/final_dataset2.nc",
    target_var="DOD",
    burn_var="burn_fraction",
    burn_threshold=0.7
):
    """
    Spatiotemporal approach:
      - Each (year, pixel) is one sample.
      - The target y is DOD[year, pixel].
      - For each predictor:
         (A) shape=(year,pixel) => used directly,
         (B) shape=(pixel,) => replicated along year.
      - BUT burned vs. unburned is defined by the *pixel’s 15-year sum*:
         if sum(burn_fraction over 15 yrs) >= threshold, => all years of that pixel = burned.
      - Train on unburned => train/test split, then test on leftover unburned + all burned.
    """

    print(f"Loading dataset: {dataset_path}")
    ds = xr.open_dataset(dataset_path)

    # 1) Gather spatiotemporal features
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    # 2) Prepare the target: shape (year, pixel)
    dod_2d = ds[target_var].values  # shape (15, pixel)

    # 3) For burn fraction, we define burned vs unburned *per pixel* based on sum across all years
    burn_2d = ds[burn_var].values  # shape (15, pixel)
    burn_sum_pixel = np.nansum(burn_2d, axis=0)  # shape (pixel,)
    pixel_burned_mask = (burn_sum_pixel >= burn_threshold)  # shape (pixel,)

    # We'll replicate pixel_burned_mask to shape (year, pixel)
    n_years = ds.dims['year']
    burned_mask_2d = np.tile(pixel_burned_mask, (n_years, 1))  # shape (year, pixel)
    unburned_mask_2d = ~burned_mask_2d

    # 4) Flatten everything => each row = (year_i, pixel_j)
    # Flatten target y
    y_all = dod_2d.ravel(order='C')  # shape (#samples,) = 15 * #pixels
    # Flatten burned mask
    burned_mask_flat   = burned_mask_2d.ravel(order='C')
    unburned_mask_flat = unburned_mask_2d.ravel(order='C')

    # Flatten each feature
    X_cols = []
    for fname in feat_names:
        arr_2d = feat_dict[fname]  # shape (year,pixel)
        arr_1d = arr_2d.ravel(order='C')  # shape (#samples,)
        X_cols.append(arr_1d)
    # stack => shape(#samples, n_features)
    X_all = np.column_stack(X_cols)

    # 5) remove any NaN from X_all or y_all
    valid_mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)

    # combine with unburned/burned
    unburned_mask_final = unburned_mask_flat & valid_mask
    burned_mask_final   = burned_mask_flat   & valid_mask

    n_total    = X_all.shape[0]
    n_unburned = np.sum(unburned_mask_final)
    n_burned   = np.sum(burned_mask_final)
    print(f"\nTotal spatiotemporal samples: {n_total}  (year*pixel)")
    print(f"  valid_mask => {np.sum(valid_mask)} used")
    print(f"  unburned => {n_unburned}")
    print(f"  burned   => {n_burned}")

    if n_unburned == 0:
        print("No unburned samples => cannot train model.")
        return

    # 6) Build unburned sets
    X_unburned = X_all[unburned_mask_final]
    y_unburned = y_all[unburned_mask_final]

    X_burned = X_all[burned_mask_final]
    y_burned = y_all[burned_mask_final]

    # 7) train/test split among unburned
    from sklearn.model_selection import train_test_split
    X_train, X_test_unburn, y_train, y_test_unburn = train_test_split(
        X_unburned, y_unburned, test_size=0.3, random_state=42
    )
    print(f"  unburned train: {len(X_train)}, unburned test: {len(X_test_unburn)}")

    # 8) Train a random forest
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    from sklearn.metrics import mean_squared_error, r2_score
    # unburned test
    y_pred_unburn = model.predict(X_test_unburn)
    mse_unburn = mean_squared_error(y_test_unburn, y_pred_unburn)
    r2_unburn  = r2_score(y_test_unburn, y_pred_unburn)
    print(f"\nUnburned Test => MSE: {mse_unburn:.4f}, R^2: {r2_unburn:.4f}")

    # burned
    if len(X_burned) > 0:
        y_pred_burn = model.predict(X_burned)
        mse_burn = mean_squared_error(y_burned, y_pred_burn)
        r2_burn  = r2_score(y_burned, y_pred_burn)
        print(f"Burned => MSE: {mse_burn:.4f}, R^2: {r2_burn:.4f}, #samples: {len(X_burned)}")

    # Feature importances
    importances = model.feature_importances_
    print("\nFeature Importances:")
    feat_names = sorted(feat_dict.keys())
    for fname, imp in zip(feat_names, importances):
        print(f"  {fname}: {imp:.4f}")


if __name__ == "__main__":
    spatiotemporal_model_with_pixelSum(
        dataset_path="/Users/yashnilmohanty/Desktop/final_dataset2.nc",
        target_var="DOD",
        burn_var="burn_fraction",
        burn_threshold=0.7
    )
