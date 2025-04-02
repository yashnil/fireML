#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

##################################################
# 1) Flatten the spatiotemporal data
##################################################
def gather_features_spatiotemporal(ds, target_var="DOD"):
    """
    For each data_var (except the target & known coords/index fields):
      - If shape=(year,pixel), we keep it as is.
      - If shape=(pixel,), replicate across 'year' dimension => shape(year,pixel).
    Returns a dict: {var_name: arr_2d(year,pixel)}
    """
    exclude_vars = {
        target_var.lower(),
        'lat', 'lon', 'latitude', 'longitude',
        'pixel', 'year', 'ncoords_vector', 'nyears_vector'
    }
    all_feats = {}

    n_years = ds.dims['year']  # e.g. 15

    for vname in ds.data_vars:
        if vname.lower() in exclude_vars:
            continue
        da = ds[vname]
        dims = set(da.dims)
        if dims == {'year','pixel'}:
            # shape (year,pixel)
            arr_2d = da.values
            all_feats[vname] = arr_2d
        elif dims == {'pixel'}:
            # shape (pixel,) => replicate across year => (year,pixel)
            arr_1d = da.values
            arr_2d = np.tile(arr_1d, (n_years,1))
            all_feats[vname] = arr_2d
        else:
            # skip unknown shapes
            pass
    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    """
    Flatten X and y from shape (year,pixel) => (#samples, #features).
    Returns X_all, y_all, feat_names, valid_mask, n_years, n_pixels.
    """
    # gather all features
    feat_dict = gather_features_spatiotemporal(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    # Flatten each feature into (#samples,) and stack
    X_cols = []
    for fname in feat_names:
        arr_2d = feat_dict[fname]   # shape (year,pixel)
        arr_1d = arr_2d.ravel(order='C')  # (#samples,)
        X_cols.append(arr_1d)
    X_all = np.column_stack(X_cols) # shape(#samples, n_features)

    # Flatten the target
    dod_2d = ds[target_var].values  # shape (year,pixel)
    y_all  = dod_2d.ravel(order='C')# shape(#samples,)

    n_years = ds.dims['year']
    n_pixels= ds.dims['pixel']

    # valid_mask => no NaN in X or y
    valid_mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)

    return X_all, y_all, feat_names, valid_mask, n_years, n_pixels

##################################################
# 2) Compute cumsum for burn_fraction => classify each sample
##################################################
def cumsum_burn_categories(ds, threshold):
    """
    For each pixel, do cumsum along year dimension => shape(year,pixel).
    Then for each (year,pixel), if cumsum >= threshold => burned, else unburned.
    Returns cat_2d with shape(year,pixel), 0=unburned,1=burned
    """
    burn_2d = ds["burn_fraction"].values  # shape(year,pixel)
    burn_cum_2d = np.cumsum(burn_2d, axis=0) # shape(year,pixel)
    cat_2d = np.where(burn_cum_2d >= threshold, 1, 0)
    return cat_2d

##################################################
# 3) Train on unburned => test on leftover unburned + burned
##################################################
def train_and_plot_cumsum(
    X_all, y_all, valid_mask, cat_2d,
    threshold=0.25
):
    """
    We interpret cat_2d: 0=unburned,1=burned for each (year,pixel) under cumsum approach
    => flatten => cat_flat, then filter with valid_mask.

    We do a train/test split on unburned samples => cat=0, then evaluate on:
      - leftover unburned test set,
      - all burned => cat=1
    """
    cat_flat = cat_2d.ravel(order='C')
    cat_valid= cat_flat[valid_mask]

    X_valid = X_all[valid_mask]
    y_valid = y_all[valid_mask]

    # define unburned => cat=0
    is_unburned = (cat_valid==0)
    n_unburned = np.sum(is_unburned)
    print(f"  threshold={threshold}, unburned samples => {n_unburned}")
    if n_unburned==0:
        print("No unburned => can't train.")
        return

    # build unburned subset
    X_unburned = X_valid[is_unburned]
    y_unburned = y_valid[is_unburned]

    # train/test split
    X_train, X_test_ub, y_train, y_test_ub = train_test_split(
        X_unburned, y_unburned, test_size=0.3, random_state=42
    )
    print(f"     unburned train={len(X_train)}, test={len(X_test_ub)}")

    # Train
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate on unburned test => scatter
    y_pred_ub = rf.predict(X_test_ub)
    plot_scatter(y_test_ub, y_pred_ub, title=f"Unburned Test (th={threshold})")

    # Evaluate on burned => cat=1
    is_burned = (cat_valid==1)
    if np.sum(is_burned)==0:
        print("No burned samples => skipping.")
        return
    X_burned = X_valid[is_burned]
    y_burned = y_valid[is_burned]

    y_pred_burn = rf.predict(X_burned)
    plot_scatter(y_burned, y_pred_burn, title=f"Burned (th={threshold})")

def plot_scatter(y_true, y_pred, title="Scatter"):
    """
    Scatter of observed (y-axis) vs predicted (x-axis).
    Show RMSE, bias, R^2.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--',label='1:1 line')

    # compute stats
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DOD")
    plt.ylabel("Observed DOD")
    plt.legend()
    plt.tight_layout()
    plt.show()

##########################
# Main
##########################
if __name__=="__main__":
    # Load the dataset
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset2.nc")
    # Flatten spatiotemporal
    X_all, y_all, feat_names, valid_mask, n_years, n_pixels = flatten_spatiotemporal(ds, "DOD")

    # We'll do 2 thresholds => 0.25, then 0.5
    for th in [0.25, 0.5]:
        print(f"\n======= CUMSUM APPROACH: threshold={th} =======")
        # define cat_2d => shape(year,pixel): 0=unburned,1=burned
        cat_2d = cumsum_burn_categories(ds, threshold=th)

        # train on unburned => test on unburned leftover + burned
        train_and_plot_cumsum(X_all, y_all, valid_mask, cat_2d, threshold=th)
