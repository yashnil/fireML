#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

########################################################
# 1) Flatten the spatiotemporal data (#samples, #features)
########################################################
def gather_features_spatiotemporal(ds, target_var="DOD"):
    """
    Collects every data_var except target and known coordinate/index fields.
    For shape (year,pixel), keep as is.
    For shape (pixel,), replicate across year => shape(year,pixel).
    Returns a dict of {var_name: arr_2d (year,pixel)}.
    """
    exclude_vars = {
        target_var.lower(), 'lat', 'lon', 'latitude', 'longitude',
        'pixel', 'year', 'ncoords_vector', 'nyears_vector'
    }
    all_feats = {}
    n_years = ds.dims['year']

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
            # replicate across year
            arr_1d = da.values
            arr_2d = np.tile(arr_1d, (n_years,1))
            all_feats[vname] = arr_2d
        else:
            # skip
            pass
    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    """
    Flatten X and y from shape (year,pixel) => (#samples, #features).
    Returns: X_all, y_all, feat_names, valid_mask, n_years, n_pixels
    """
    feat_dict = gather_features_spatiotemporal(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    # Flatten each feature
    X_cols = []
    for fname in feat_names:
        arr_2d = feat_dict[fname]  # shape (year,pixel)
        arr_1d = arr_2d.ravel(order='C') # (#samples,)
        X_cols.append(arr_1d)
    X_all = np.column_stack(X_cols)

    # Flatten DOD
    dod_2d = ds[target_var].values  # shape (year,pixel)
    y_all = dod_2d.ravel(order='C')  # (#samples,)

    # Valid mask => no NaN in features or target
    valid_mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)

    n_years = ds.dims['year']
    n_pixels= ds.dims['pixel']
    return X_all, y_all, feat_names, valid_mask, n_years, n_pixels

########################################################
# 2) Define 4 categories based on pixel-level cumulative burn
########################################################
def define_4cats_static(ds):
    """
    Summation of burn_fraction over 15 years => shape(pixel,).
    If sum < 0.25 => cat=0
    If 0.25..0.5 => cat=1
    If 0.5..0.75 => cat=2
    If >=0.75 => cat=3
    Replicate each pixel's category for all 15 years => cat_2d (year,pixel).
    """
    burn_2d = ds["burn_fraction"].values  # shape(year,pixel)
    burn_sum_pix = np.nansum(burn_2d, axis=0)  # shape(pixel,)

    n_pixels = ds.dims['pixel']
    pixel_cat = np.zeros(n_pixels, dtype=int)
    for i in range(n_pixels):
        val = burn_sum_pix[i]
        if val < 0.25:
            pixel_cat[i] = 0
        elif val < 0.5:
            pixel_cat[i] = 1
        elif val < 0.75:
            pixel_cat[i] = 2
        else:
            pixel_cat[i] = 3

    n_years = ds.dims['year']
    cat_2d = np.tile(pixel_cat, (n_years,1))  # shape(year,pixel)
    return cat_2d

########################################################
# 3) Train on "unburned" subset => separate runs for threshold=0.25 vs threshold=0.5
########################################################
def run_spatiotemporal_experiment(
    ds, X_all, y_all, feat_names, valid_mask,
    cat_2d, # shape(year,pixel) => 4 categories (0..3)
    unburned_max_cat=0
):
    """
    unburned_max_cat=0 => means only cat=0 is used for training (0..0.25)
    unburned_max_cat=1 => means cat=0 or cat=1 => sum <0.5 is used for training
    Then we do train/test split among that subset, test on cat=0,1,2,3 => scatterplots
    """
    # Flatten cat
    cat_flat = cat_2d.ravel(order='C')
    cat_valid = cat_flat[valid_mask]
    X_valid   = X_all[valid_mask]
    y_valid   = y_all[valid_mask]

    # define "unburned" mask among valid samples
    # if unburned_max_cat=0 => unburned => cat=0
    # if unburned_max_cat=1 => unburned => cat in [0,1]
    is_unburned = (cat_valid <= unburned_max_cat)
    n_unburned = np.sum(is_unburned)
    print(f"Training threshold => unburned up to cat={unburned_max_cat} => #unburned={n_unburned}")

    if n_unburned==0:
        print("No unburned samples => can't train.")
        return

    X_unburn = X_valid[is_unburned]
    y_unburn = y_valid[is_unburned]

    # train/test split
    X_train, X_test_u, y_train, y_test_u = train_test_split(
        X_unburn, y_unburn, test_size=0.3, random_state=42
    )
    print(f"   unburned train={len(X_train)}, test={len(X_test_u)}")

    # train a random forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on unburned test => cat= ??? => cat <= unburned_max_cat
    y_pred_u = model.predict(X_test_u)
    plot_scatter(y_test_u, y_pred_u, "Unburned Test")

    # Evaluate on each cat=0..3 => scatter
    for cval in [0,1,2,3]:
        cat_mask = (cat_valid==cval)
        if np.sum(cat_mask)==0:
            continue
        X_cat = X_valid[cat_mask]
        y_cat = y_valid[cat_mask]
        y_pred_cat = model.predict(X_cat)
        label_str = f"Cat={cval}" # or a descriptive label
        plot_scatter(y_cat, y_pred_cat, label_str)


def plot_scatter(y_true, y_pred, title="Scatter"):
    """
    Plot predicted vs. observed with 1:1 line, show RMSE, bias, R^2
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--',label='1:1 line')
    # metrics
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true,y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DOD")
    plt.ylabel("Observed DOD")
    plt.legend()
    plt.tight_layout()
    plt.show()

############################
# Main
############################
if __name__=="__main__":
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset2.nc")

    # Flatten spatiotemporal
    X_all, y_all, feat_names, valid_mask, n_years, n_pixels = flatten_spatiotemporal(ds, target_var="DOD")

    # define 4 categories => cat=0 => [0..0.25), cat=1 => [0.25..0.5), cat=2 => [0.5..0.75), cat=3 => [0.75..999]
    cat_2d = define_4cats_static(ds) # shape(year,pixel)

    print("==== RUN #1 => unburned up to 0.25 (cat=0) ====")
    run_spatiotemporal_experiment(ds, X_all, y_all, feat_names, valid_mask, cat_2d, unburned_max_cat=0)

    print("\n==== RUN #2 => unburned up to 0.50 (cat=0 or cat=1) ====")
    run_spatiotemporal_experiment(ds, X_all, y_all, feat_names, valid_mask, cat_2d, unburned_max_cat=1)
