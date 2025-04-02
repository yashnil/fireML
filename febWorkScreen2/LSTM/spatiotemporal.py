#!/usr/bin/env python3
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

########################
# 0) LOAD COORDINATES
########################
# shape(n_pixels,2): Each row is [lat, lon] for that pixel
from obtainCoordinates import coords  

########################
# 1) Pre-2004 Burn
########################
def compute_pre2004_burn(coords, path_pattern, year_start=2001, year_end=2003):
    """
    Summation from 2001..2003 for each pixel bounding box.
    """
    import xarray as xr  # local import for clarity
    n_pixels = len(coords)
    pre_burn = np.zeros((n_pixels,), dtype=np.float32)

    years = range(year_start, year_end+1)
    months= range(1,13)

    for yr in years:
        for mm in months:
            file_path = path_pattern.format(year=yr, month=mm)
            ds_nc = xr.open_dataset(file_path)
            burn_2d = ds_nc["MTBS_BurnFraction"].values
            lat_2d  = ds_nc["XLAT_M"].values
            lon_2d  = ds_nc["XLONG_M"].values

            # Flatten
            flat_burn = burn_2d.ravel()
            flat_lat  = lat_2d.ravel()
            flat_lon  = lon_2d.ravel()

            # Clamp fraction>1 =>1
            flat_burn = np.minimum(flat_burn, 1.0)

            # Remove NaNs
            valid_mask = (
                np.isfinite(flat_burn) &
                np.isfinite(flat_lat) &
                np.isfinite(flat_lon)
            )
            fb  = flat_burn[valid_mask]
            fla = flat_lat[valid_mask]
            flo = flat_lon[valid_mask]

            # Accumulate for each pixel bounding box
            for i, (coord_lat, coord_lon) in enumerate(coords):
                lat_min, lat_max = coord_lat - 0.005, coord_lat + 0.005
                lon_min, lon_max = coord_lon - 0.005, coord_lon + 0.005
                in_box = (
                    (fla >= lat_min) & (fla <= lat_max) &
                    (flo >= lon_min) & (flo <= lon_max)
                )
                box_burn = fb[in_box]
                if len(box_burn) > 0:
                    mean_frac = np.mean(box_burn)
                else:
                    mean_frac = 0.0
                pre_burn[i] += mean_frac

            ds_nc.close()

    # Final clamp => <=1.0 if desired
    pre_burn = np.minimum(pre_burn, 1.0)
    return pre_burn

########################
# 2) cumsum with initial
########################
def compute_burn_cumsum_with_initial(ds, pre_burn):
    burn_2d = ds["burn_fraction"].values  # shape(15, pixel)
    n_years, n_pixels = burn_2d.shape
    cumsum_2d = np.zeros((n_years, n_pixels), dtype=np.float32)

    # year=0 => 2004
    cumsum_2d[0, :] = pre_burn + burn_2d[0, :]

    for y in range(1, n_years):
        cumsum_2d[y, :] = cumsum_2d[y - 1, :] + burn_2d[y, :]

    return cumsum_2d

########################
# 3) Gather features
########################
def gather_spatiotemporal_features(ds, target_var="DOD"):
    """
    Collects all data_vars except the target, lat, lon, etc.
    Returns a dict of {varName -> 2D array (year, pixel)}.
    If the variable is only (pixel,) we tile it across years.
    """
    exclude_vars = {
        target_var.lower(),
        'lat','lon','latitude','longitude',
        'pixel','year','ncoords_vector','nyears_vector'
    }
    all_feats = {}
    ny = ds.dims['year']

    for var in ds.data_vars:
        if var.lower() in exclude_vars:
            continue
        da = ds[var]
        dims = set(da.dims)
        if dims == {'year', 'pixel'}:
            # 2D variable => shape (year, pixel)
            arr2d = da.values
            all_feats[var] = arr2d
        elif dims == {'pixel'}:
            # 1D variable => replicate across all years
            arr1d = da.values
            arr2d = np.tile(arr1d, (ny,1))
            all_feats[var] = arr2d

    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    """
    Creates a single large X array of shape (N, n_features),
    and a y array of shape (N,),
    where N = year * pixel.
    valid_mask indicates which rows are free of NaN.
    """
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    X_cols = []
    for f in feat_names:
        arr2d = feat_dict[f]  # shape(year, pixel)
        arr1d = arr2d.ravel(order='C')  # flatten
        X_cols.append(arr1d)

    X_all = np.column_stack(X_cols)

    # Flatten target
    dod2d = ds[target_var].values  # shape(year, pixel)
    y_all = dod2d.ravel(order='C')

    # valid mask
    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_all)
    )

    n_years = ds.dims['year']
    n_pixels = ds.dims['pixel']
    return X_all, y_all, feat_names, valid_mask, n_years, n_pixels

########################
# 4) define c0..c3
########################
def define_4cats_cumsum(cumsum_2d):
    cat_2d = np.zeros(cumsum_2d.shape, dtype=int)
    c0 = cumsum_2d < 0.25
    c1 = (cumsum_2d >= 0.25) & (cumsum_2d < 0.5)
    c2 = (cumsum_2d >= 0.5) & (cumsum_2d < 0.75)
    c3 = (cumsum_2d >= 0.75)
    cat_2d[c0] = 0
    cat_2d[c1] = 1
    cat_2d[c2] = 2
    cat_2d[c3] = 3
    return cat_2d

########################
# 5) scatter
########################
def plot_scatter(y_true, y_pred, title="Scatter"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title="Scatter by Category"):
    """
    Color‐code each sample by its category (0..3).
    Adds a 1:1 line + legend, and shows RMSE/bias/R².
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    plt.figure(figsize=(6,6))

    # Choose colors for each category
    cat_colors = {0:'red', 1:'green', 2:'blue', 3:'orange'}

    # Plot points category by category
    for cval in cat_colors.keys():
        mask = (cat == cval)
        if np.any(mask):
            plt.scatter(
                y_pred[mask],
                y_true[mask],
                alpha=0.4,
                color=cat_colors[cval],
                label=f"cat={cval}"
            )

    # 1:1 line
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

########################
# BIAS HIST
########################
def plot_bias_hist(y_true, y_pred, title="Bias Histogram", x_min=-100, x_max=300):
    residuals = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=50, range=(x_min,x_max), alpha=0.7)

    mean_bias = np.mean(residuals)
    std_bias  = np.std(residuals)

    plt.axvline(mean_bias, color='k', linestyle='dashed', linewidth=2)
    plt.title(f"{title}\nMean={mean_bias:.2f}, Std={std_bias:.2f}")
    plt.xlabel("Bias (Pred - Obs)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

########################
# ELEV + VEG => one boxplot
########################
def plot_boxplot_dod_by_elev_veg(y_dod, elev, vegtyp, cat_label="c0"):
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin_idx = np.digitize(elev, elev_edges) - 1
    n_elev_bins = len(elev_edges) - 1

    unique_veg = np.unique(vegtyp)

    box_data = []
    x_labels = []
    for elev_i in range(n_elev_bins):
        for veg_v in unique_veg:
            sel = (
                (elev_bin_idx == elev_i) &
                (vegtyp == veg_v)
            )
            y_sel = y_dod[sel]
            if len(y_sel) == 0:
                y_sel = np.array([np.nan])  # so boxplot doesn't break
            box_data.append(y_sel)
            low  = elev_edges[elev_i]
            high = elev_edges[elev_i + 1]
            elev_label = f"E[{int(low)}-{int(high)}]"
            veg_label  = f"Veg={veg_v}"
            x_labels.append(f"{elev_label}, {veg_label}")

    plt.figure(figsize=(12,5))
    plt.boxplot(box_data, showmeans=True)
    plt.xticks(range(1, len(box_data)+1), x_labels, rotation=90)
    plt.xlabel("(ElevationBin, VegTyp)")
    plt.ylabel("Raw DoD")
    plt.title(f"{cat_label}: DoD vs. (Elev, Veg)")
    plt.tight_layout()
    plt.show()

########################
# Permutation Importance for LSTM
########################
def compute_permutation_importance_lstm(
    model, 
    X_val,    # shape (N, n_features)
    y_val, 
    batch_size=512, 
    n_repeats=3, 
    random_state=42
):
    import numpy as np
    from sklearn.metrics import mean_squared_error

    rng = np.random.RandomState(random_state)

    # First reshape X_val so that it matches what the LSTM expects
    N, n_features = X_val.shape
    X_val_3D = X_val.reshape(N, 1, n_features)

    # --- Baseline prediction and MSE ---
    y_pred = model.predict(X_val_3D, batch_size=batch_size).squeeze()
    baseline_mse = mean_squared_error(y_val, y_pred)

    importances = np.zeros(n_features, dtype=float)

    # We'll shuffle each feature in the 2D form,
    # but re-reshape before calling predict.
    X_val_perm = np.copy(X_val)
    for f_idx in range(n_features):
        scores = []
        for _ in range(n_repeats):
            # Shuffle this feature in X_val_perm
            saved_col = np.copy(X_val_perm[:, f_idx])
            perm_idx = rng.permutation(N)
            X_val_perm[:, f_idx] = X_val_perm[perm_idx, f_idx]

            # Reshape to 3D, predict
            X_val_perm_3D = X_val_perm.reshape(N, 1, n_features)
            y_pred_p = model.predict(X_val_perm_3D, batch_size=batch_size).squeeze()
            perm_mse = mean_squared_error(y_val, y_pred_p)
            scores.append(perm_mse)

            # revert the shuffled column
            X_val_perm[:, f_idx] = saved_col

        importances[f_idx] = np.mean(scores) - baseline_mse

    return importances

########################
# Feature Importance Plot
########################
def plot_top10_features(feat_importances, feat_names, title="Top 10 Feature Importances"):
    importances = np.array(feat_importances)
    idx_sorted = np.argsort(importances)[::-1]  # descending
    top10_idx  = idx_sorted[:10]
    top10_vals = importances[top10_idx]
    top10_names= [feat_names[i] for i in top10_idx]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(top10_vals)), top10_vals, align='center')
    plt.xticks(range(len(top10_vals)), top10_names, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Permutation Importance\n(Δ MSE vs. baseline)")
    plt.tight_layout()
    plt.show()

########################
# Plot top-5 feature-value vs. DOD
########################
def plot_top5_feature_scatter(feat_importances, X_valid_raw, y_valid_raw, cat_valid,
                              feat_names, title_prefix="(thr=0)"):
    """
    Takes the raw (unscaled) X_valid, y_valid, identifies top-5 features by the
    permutation importances array, then scatterplots feature vs. DoD (colored by category).
    """
    importances = np.array(feat_importances)
    idx_sorted = np.argsort(importances)[::-1]
    top5_idx   = idx_sorted[:5]

    cat_colors = {0:'red', 1:'blue', 2:'green', 3:'purple'}

    for feat_idx in top5_idx:
        fname = feat_names[feat_idx]
        feat_vals = X_valid_raw[:, feat_idx]

        plt.figure(figsize=(6,5))
        for cval, ccolor in cat_colors.items():
            sel_c = (cat_valid == cval)
            plt.scatter(
                y_valid_raw[sel_c],
                feat_vals[sel_c],
                c=ccolor, alpha=0.4, label=f"cat={cval}"
            )
        # line of best fit => across ALL data
        x_all = y_valid_raw
        y_all = feat_vals
        mask_lin = np.isfinite(x_all) & np.isfinite(y_all)
        if np.sum(mask_lin) > 2:
            x_lin = x_all[mask_lin]
            y_lin = y_all[mask_lin]
            p = np.polyfit(x_lin, y_lin, 1)  # slope, intercept
            x_min, x_max = np.min(x_lin), np.max(x_lin)
            x_vals = np.linspace(x_min, x_max, 100)
            y_fit  = np.polyval(p, x_vals)
            plt.plot(x_vals, y_fit, 'k--', label=f"Best fit (y={p[0]:.2f}x+{p[1]:.2f})")

        plt.xlabel("Observed DOD (raw)")
        plt.ylabel(f"{fname} (raw)")
        plt.title(f"{title_prefix}: Feature={fname}")
        plt.legend()
        plt.tight_layout()
        plt.show()

########################
# LSTM experiment w/ scaling
########################
def run_spatiotemporal_experiment_lstm(
    X_all_raw, y_all_raw, valid_mask, cat_2d,
    ds=None, feat_names=None,
    unburned_max_cat=0,
    epochs=100, batch_size=512
):
    """
    Trains an LSTM on the "unburned" subset (cat <= unburned_max_cat),
    using standard scaling. Then evaluates on each category (0..3) + all data,
    producing scatter/hist/boxplots and permutation-based feature importance.

    Because we flatten year/pixel to one sample, each sample is just (1, n_features).
    That means we don't have a real time-series dimension. The LSTM effectively
    acts like a small MLP. But this code structure still works so you can
    later adapt it to multi-year sequences if desired.
    """

    # 1) Subset "valid"
    cat_flat  = cat_2d.ravel(order='C')
    cat_valid = cat_flat[valid_mask]   # shape (#valid,)
    X_valid_raw = X_all_raw[valid_mask] # shape (#valid, n_features)
    y_valid_raw = y_all_raw[valid_mask]

    # For boxplots:
    elev_2d = ds["Elevation"].values
    veg_2d  = ds["VegTyp"].values
    elev_1d_full = elev_2d.ravel(order='C')
    veg_1d_full  = veg_2d.ravel(order='C')
    elev_valid  = elev_1d_full[valid_mask]
    veg_valid   = veg_1d_full[valid_mask]

    # 2) Subset "unburned" => cat <= unburned_max_cat
    is_unburned = (cat_valid <= unburned_max_cat)
    if not np.any(is_unburned):
        print(f"No unburned samples for cat <= {unburned_max_cat}. Cannot train.")
        return None
    X_ub_raw = X_valid_raw[is_unburned]
    y_ub_raw = y_valid_raw[is_unburned]

    # 3) Train/Val split (on unburned)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_ub_raw, y_ub_raw, test_size=0.3, random_state=42
    )
    print(f" [LSTM] cat <= {unburned_max_cat}, #train={len(X_train_raw)}, #test={len(X_test_raw)}")

    # 4) Scale X and y
    #    Fit scalers on TRAIN, apply to both TRAIN and TEST,
    #    then also to "valid" for final predictions.
    xscaler = StandardScaler()
    yscaler = StandardScaler()

    X_train_sc = xscaler.fit_transform(X_train_raw)
    y_train_sc = yscaler.fit_transform(y_train_raw.reshape(-1,1)).ravel()

    X_test_sc  = xscaler.transform(X_test_raw)
    y_test_sc  = yscaler.transform(y_test_raw.reshape(-1,1)).ravel()

    X_valid_sc = xscaler.transform(X_valid_raw)
    # We do NOT transform y_valid for final scatter yet, but for predictions we do:
    # We'll invert the predictions later so we can compare on the raw scale.

    # 5) Reshape to (N, 1, n_features) for LSTM
    n_features = X_train_sc.shape[1]
    X_train_3D = X_train_sc.reshape((X_train_sc.shape[0], 1, n_features))
    X_test_3D  = X_test_sc.reshape((X_test_sc.shape[0], 1, n_features))
    X_valid_3D = X_valid_sc.reshape((X_valid_sc.shape[0], 1, n_features))

    # 6) Build & train LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential()
    model.add(LSTM(32, input_shape=(1, n_features)))
    model.add(Dense(1, activation='linear'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    history = model.fit(
        X_train_3D, y_train_sc,
        validation_data=(X_test_3D, y_test_sc),
        epochs=epochs, batch_size=batch_size,
        verbose=1
    )

    # ========== Evaluate on train (unburned) ==========
    y_pred_train_sc = model.predict(X_train_3D, batch_size=batch_size).squeeze()
    # invert scale
    y_pred_train = yscaler.inverse_transform(y_pred_train_sc.reshape(-1,1)).ravel()
    # Plot
    plot_scatter(y_train_raw, y_pred_train,
        title=f"LSTM Unburned Train (cat<={unburned_max_cat})")
    plot_bias_hist(y_train_raw, y_pred_train,
        title=f"LSTM Bias Hist: Unburned Train (cat<={unburned_max_cat})")

    # ========== Evaluate on test (unburned) ==========
    y_pred_test_sc = model.predict(X_test_3D, batch_size=batch_size).squeeze()
    y_pred_test = yscaler.inverse_transform(y_pred_test_sc.reshape(-1,1)).ravel()

    plot_scatter(y_test_raw, y_pred_test,
        title=f"LSTM Unburned Test (cat<={unburned_max_cat})")
    plot_bias_hist(y_test_raw, y_pred_test,
        title=f"LSTM Bias Hist: Unburned Test (cat<={unburned_max_cat})")

    # ========== Evaluate on each cat=0..3 (all valid) ==========
    for cval in [0,1,2,3]:
        sel_cat = (cat_valid == cval)
        if not np.any(sel_cat):
            print(f"No samples cat={cval}.")
            continue

        X_c_raw = X_valid_raw[sel_cat]
        y_c_raw = y_valid_raw[sel_cat]

        X_c_sc = xscaler.transform(X_c_raw)
        X_c_3D = X_c_sc.reshape((X_c_sc.shape[0], 1, n_features))

        y_pred_c_sc = model.predict(X_c_3D, batch_size=batch_size).squeeze()
        y_pred_c     = yscaler.inverse_transform(y_pred_c_sc.reshape(-1,1)).ravel()

        plot_scatter(y_c_raw, y_pred_c, title=f"LSTM Category={cval}")
        plot_bias_hist(y_c_raw, y_pred_c, title=f"LSTM Bias Hist: cat={cval}")

        # Boxplot
        cat_label = f"cat={cval}, thr={unburned_max_cat}"
        plot_boxplot_dod_by_elev_veg(y_c_raw, elev_valid[sel_cat], veg_valid[sel_cat],
                                     cat_label=cat_label)

    # ========== Evaluate on ALL valid data ==========
    y_pred_all_sc = model.predict(X_valid_3D, batch_size=batch_size).squeeze()
    y_pred_all = yscaler.inverse_transform(y_pred_all_sc.reshape(-1,1)).ravel()

    plot_scatter_by_cat(y_valid_raw, y_pred_all, cat_valid,
                    title=f"LSTM All Data (thr={unburned_max_cat})")
    plot_bias_hist(y_valid_raw, y_pred_all,
                   title=f"LSTM Bias Hist: All Data (thr={unburned_max_cat})")

    # ========== Permutation-based feature importance ==========
    # We'll compute it on the unburned TEST subset for simplicity:
    # Use the scaled test set
    perm_importances = compute_permutation_importance_lstm(
        model,
        X_test_sc,  # shape (N, n_features) in scaled space
        y_test_sc,
        batch_size=batch_size
    )

    plot_top10_features(
        perm_importances,
        feat_names,
        title=f"LSTM Top 10 Features (thr={unburned_max_cat})"
    )

    # top-5 feature-value vs. DOD (in raw space, color-coded by cat)
    cat_flat_all = cat_2d.ravel(order='C')[valid_mask]
    plot_top5_feature_scatter(
        perm_importances,
        X_valid_raw,  # raw
        y_valid_raw,  # raw
        cat_flat_all,
        feat_names,
        title_prefix=f"(thr={unburned_max_cat})"
    )

    return model

###############################
# MAIN
###############################
if __name__=="__main__":

    # 1) LOAD/COMPUTE PRE-BURN
    path_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc"

    pre_burn  = compute_pre2004_burn(coords, path_pat, 2001, 2003)

    # 2) LOAD DS
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset2.nc")

    # 3) cumsum burn fraction
    cumsum_2d = compute_burn_cumsum_with_initial(ds, pre_burn)
    ds["burn_cumsum"] = (("year","pixel"), cumsum_2d)

    # 4) define 4 categories
    cat_2d = define_4cats_cumsum(cumsum_2d)

    # 5) Flatten => X_all, y_all, feat_names
    X_all, y_all, feat_names, valid_mask, n_years, n_pixels = flatten_spatiotemporal(ds, "DOD")

    # 6) RUN #1 => thr=0 => unburned cat=0
    print("\n==== LSTM RUN #1 => unburned= cat=0 (<0.25) ====")
    lstm_run1 = run_spatiotemporal_experiment_lstm(
        X_all, y_all, valid_mask, cat_2d,
        ds=ds, feat_names=feat_names,
        unburned_max_cat=0,   # cat=0 is considered unburned
        epochs=100,            # more epochs to ensure better convergence
        batch_size=512
    )

    # 7) RUN #2 => thr=0.5 => unburned cat=0..1
    print("\n==== LSTM RUN #2 => unburned= cat=0,1 (<0.5) ====")
    lstm_run2 = run_spatiotemporal_experiment_lstm(
        X_all, y_all, valid_mask, cat_2d,
        ds=ds, feat_names=feat_names,
        unburned_max_cat=1,
        epochs=100,
        batch_size=512
    )

    print("DONE.")
