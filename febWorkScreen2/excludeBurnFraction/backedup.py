#!/usr/bin/env python3
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import ranksums  # for Wilcoxon rank-sum tests
import numpy.random as npr  # for random subsampling in the final step
from typing import List, Tuple, Optional

############################################################
# 0) LOAD COORDINATES
############################################################
from obtainCoordinates import coords  # shape(n_pixels,2)

############################################################
# 1) Pre-2004 Burn
############################################################
def compute_pre2004_burn(coords, path_pattern, year_start=2001, year_end=2003):
    """
    Summation from 2001..2003 for each pixel bounding box.
    """
    import xarray as xr
    n_pixels = len(coords)
    pre_burn = np.zeros((n_pixels,), dtype=np.float32)

    years = range(year_start, year_end + 1)
    months= range(1, 13)

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

            # Clamp fraction>1 => 1
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
                mean_frac = np.mean(box_burn) if len(box_burn) > 0 else 0.0
                pre_burn[i] += mean_frac

            ds_nc.close()

    pre_burn = np.minimum(pre_burn, 1.0)
    return pre_burn

############################################################
# 2) cumsum with initial
############################################################
def compute_burn_cumsum_with_initial(ds, pre_burn):
    """
    Add pre_burn to year=0, then cumsum across subsequent years,
    for defining categories only (not a predictor).
    """
    burn_2d = ds["burn_fraction"].values
    n_years, n_pixels = burn_2d.shape
    cumsum_2d = np.zeros((n_years, n_pixels), dtype=np.float32)

    cumsum_2d[0, :] = pre_burn + burn_2d[0, :]
    for y in range(1, n_years):
        cumsum_2d[y, :] = cumsum_2d[y - 1, :] + burn_2d[y, :]

    return cumsum_2d

############################################################
# 3) Gather features
#    Exclude 'burn_fraction' and 'burn_cumsum'
############################################################
def gather_spatiotemporal_features(ds, target_var="DOD"):
    """
    Exclude 'burn_fraction' and 'burn_cumsum' from the predictor set.
    """
    exclude_vars = {
        target_var.lower(),
        'lat','lon','latitude','longitude',
        'pixel','year','ncoords_vector','nyears_vector',
        'burn_fraction',
        'burn_cumsum'
    }
    all_feats = {}
    n_years = ds.dims['year']

    for var_name in ds.data_vars:
        # skip if var_name.lower() in the exclude set
        if var_name.lower() in exclude_vars:
            continue

        da = ds[var_name]
        dims = set(da.dims)
        if dims == {'year', 'pixel'}:
            arr2d = da.values
            all_feats[var_name] = arr2d
        elif dims == {'pixel'}:
            arr1d = da.values
            arr2d = np.tile(arr1d, (n_years,1))
            all_feats[var_name] = arr2d

    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    X_cols = []
    for fname in feat_names:
        arr2d = feat_dict[fname]
        arr1d = arr2d.ravel(order='C')
        X_cols.append(arr1d)
    X_all = np.column_stack(X_cols)

    dod_2d = ds[target_var].values
    y_all  = dod_2d.ravel(order='C')

    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_all)
    )
    return X_all, y_all, feat_names, valid_mask

############################################################
# 4) define c0..c3 from cumsum
############################################################
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

############################################################
# 5) Plotting (scatter, hist, cat)
############################################################
def plot_scatter(y_true, y_pred, title="Scatter"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, Bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bias_hist(y_true, y_pred, title="Bias Histogram", x_min=-100, x_max=100):
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

def plot_scatter_by_cat(y_true, y_pred, cat, title="Scatter by Category"):
    plt.figure(figsize=(6,6))
    cat_colors = {0:'red', 1:'green', 2:'blue', 3:'orange'}

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
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, Bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────
# 6‑A) **NEW** simple grid maps  (bias & mean DoD)
# ──────────────────────────────────────────────────────────
from matplotlib.colors import TwoSlopeNorm     # used in both helpers

def produce_pixel_bias_map_simple(ds, pixel_idx, y_true, y_pred,
                                  lat_var="latitude", lon_var="longitude",
                                  title="Pixel Bias: Simple Grid"):
    """
    Scatter of (lon, lat) coloured by mean bias (Pred‑Obs) per pixel.
    No background tiles.  Handles the edge‑case where all biases are zero.
    """
    n_pix = ds.dims["pixel"]
    sum_b, cnt = np.zeros(n_pix), np.zeros(n_pix)

    for px, res in zip(pixel_idx, y_pred - y_true):
        sum_b[px] += res
        cnt[px]   += 1

    mean_b = np.full(n_pix, np.nan)
    mask   = cnt > 0
    mean_b[mask] = sum_b[mask] / cnt[mask]

    # 1‑D coordinates
    lat = ds[lat_var].values
    lon = ds[lon_var].values
    lat1d = lat[0] if lat.ndim == 2 else lat
    lon1d = lon[0] if lon.ndim == 2 else lon

    # colour scaling – ensure vmax > 0 to satisfy TwoSlopeNorm
    vmax = np.nanmax(np.abs(mean_b))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6                          # tiny positive number
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    plt.figure(figsize=(7,6))
    plt.scatter(lon1d, lat1d, c="lightgray", s=5, alpha=0.6)
    sc = plt.scatter(lon1d[mask], lat1d[mask], c=mean_b[mask],
                     cmap="bwr", norm=norm, s=10)
    plt.colorbar(sc, shrink=0.8, label="Mean Bias (Pred‑Obs)")
    plt.xlabel("Longitude");  plt.ylabel("Latitude");  plt.title(title)
    plt.tight_layout();  plt.show()

def plot_mean_dod_map_simple(ds, mean_vals,
                             lat_var="latitude", lon_var="longitude",
                             title="Mean DoD"):
    """Scatter map of mean DoD."""
    lat = ds[lat_var].values
    lon = ds[lon_var].values
    lat1d = lat[0] if lat.ndim == 2 else lat
    lon1d = lon[0] if lon.ndim == 2 else lon
    plt.figure(figsize=(7,6))
    sc = plt.scatter(lon1d, lat1d, c=mean_vals, cmap="viridis", s=10)
    plt.colorbar(sc, shrink=0.8, label="Mean DoD (days)")
    plt.xlabel("Longitude");  plt.ylabel("Latitude");  plt.title(title)
    plt.tight_layout();  plt.show()

def plot_boxplot_dod_by_elev_veg(y_dod, elev, vegtyp, cat_label="c0"):
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin_idx = np.digitize(elev, elev_edges) - 1
    n_elev_bins = len(elev_edges) - 1

    unique_veg = np.unique(vegtyp)

    box_data = []
    x_labels = []
    for elev_i in range(n_elev_bins):
        for veg_v in unique_veg:
            sel = (elev_bin_idx == elev_i) & (vegtyp == veg_v)
            y_sel = y_dod[sel]
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

############################################################
# 7) Feature Importance Plot (Random Forest)
############################################################
def plot_top10_features(rf_model, feat_names, title="Top 10 Feature Importances"):
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top10_idx  = idx_sorted[:10]
    top10_vals = importances[top10_idx]
    top10_names= [feat_names[i] for i in top10_idx]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(top10_vals)), top10_vals, align='center')
    plt.xticks(range(len(top10_vals)), top10_names, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_top5_feature_scatter(rf_model, X_valid, y_valid, cat_valid, feat_names,
                              title_prefix="(NoBurnFrac)"):
    """
    Show top-5 features, scatter vs. Observed DOD, color-coded by category
    """
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top5_idx   = idx_sorted[:5]

    cat_colors = {0:'red', 1:'blue', 2:'green', 3:'purple'}

    for feat_idx in top5_idx:
        fname = feat_names[feat_idx]
        feat_vals = X_valid[:, feat_idx]

        plt.figure(figsize=(6,5))
        for cval, ccolor in cat_colors.items():
            sel_c = (cat_valid == cval)
            plt.scatter(
                y_valid[sel_c],
                feat_vals[sel_c],
                c=ccolor,
                alpha=0.4,
                s=10,
                label=f"cat={cval}"
            )

        mask_lin = np.isfinite(y_valid) & np.isfinite(feat_vals)
        if np.sum(mask_lin) > 2:
            r_val = np.corrcoef(y_valid[mask_lin], feat_vals[mask_lin])[0,1]
        else:
            r_val = np.nan

        plt.legend(title=f"r={r_val:.2f}", loc="best")
        plt.xlabel("Observed DOD")
        plt.ylabel(fname)
        plt.title(f"{title_prefix}: Feature={fname}")
        plt.tight_layout()
        plt.show()

# ──────────────────────────────────────────────────────────
# 8‑A) **NEW** metrics for arbitrary burn‑fraction bins
# ──────────────────────────────────────────────────────────
def evaluate_metrics_per_bin(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              burn_vals: np.ndarray,
                              bins: List[Tuple[float, Optional[float]]],
                              label: str = "BurnFrac") -> None:
    """Print RMSE / Bias / R² for each (lo, hi] bin."""
    for lo, hi in bins:
        if hi is None:
            sel = burn_vals > lo;         tag = f">{lo*100:.0f}%"
        else:
            sel = (burn_vals >= lo) & (burn_vals < hi)
            tag = f"{lo*100:.0f}-{hi*100:.0f}%"
        n = sel.sum()
        if n == 0:
            print(f"{label} {tag:>7}: N=0 – skipped");  continue
        rmse = np.sqrt(mean_squared_error(y_true[sel], y_pred[sel]))
        bias = (y_pred[sel] - y_true[sel]).mean()
        r2   = r2_score(y_true[sel], y_pred[sel]) if n > 1 else np.nan
        print(f"{label} {tag:>7}: N={n:5d}  RMSE={rmse:6.2f}  "
              f"Bias={bias:7.2f}  R²={r2:6.3f}")

############################################################
# 8) Random Forest Experiment (NoBurnFrac)
#    70/30 split per category
#    Add rank-sum tests, downsampling, saving .txt data
############################################################
def run_rf_excluding_burnfraction(
        X_all: np.ndarray,
        y_all: np.ndarray,
        cat_2d: np.ndarray,
        valid_mask: np.ndarray,
        ds: xr.Dataset,
        feat_names: list[str]
    ):
    """
    Train / evaluate a Random‑Forest **without** using `burn_fraction`
    (it was excluded upstream) but with the full evaluation suite:

    • 70/30 split within each cumulative‑burn category (c0–c3)  
    • Standard scatter & bias‑hist plots  
    • Wilcoxon rank‑sum tests (c1–c3 vs. c0)  
    • Down‑sampling robustness check  
    • Saves per‑category obs / pred to .txt  
    • 25 %‑bin scatter plot (handled elsewhere)  
    • **NEW:** RMSE / Bias / R² for 10 % burn‑fraction bins (eval‑only)  
    • **NEW:** Lightweight bias map & mean‑DoD maps (no Cartopy tiles)  
    • Feature‑importance & top‑5 feature scatter plots
    """

    # ── 0.  Flatten categories & select valid rows ───────────────────
    cat_flat   = cat_2d.ravel(order="C")
    cat_valid  = cat_flat[valid_mask]

    X_valid, y_valid = X_all[valid_mask], y_all[valid_mask]

    # ── 1.  70 / 30 train‑test split **within each cat** ─────────────
    train_idx, test_idx = [], []
    for c in (0, 1, 2, 3):
        rows = np.where(cat_valid == c)[0]
        if rows.size == 0:
            print(f"cat={c}: no valid rows – skipped")
            continue
        tr, te = train_test_split(rows, test_size=0.3, random_state=42)
        train_idx.append(tr);  test_idx.append(te)

    if not train_idx:
        print("No training data available – abort.");  return None

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    X_train, y_train = X_valid[train_idx], y_valid[train_idx]
    X_test,  y_test  = X_valid[test_idx],  y_valid[test_idx]

    # ── 2.  Train RF ─────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ── 3.  Basic plots (train / test) ───────────────────────────────
    y_pred_train = rf.predict(X_train)
    plot_scatter(y_train, y_pred_train,
                 title="RF (NoBurnFrac): Train (70 % each cat)")
    plot_bias_hist(y_train, y_pred_train,
                   title="RF (NoBurnFrac): Bias Hist (Train)")

    y_pred_test = rf.predict(X_test)
    plot_scatter(y_test, y_pred_test,
                 title="RF (NoBurnFrac): Test (30 % each cat)")
    plot_bias_hist(y_test, y_pred_test,
                   title="RF (NoBurnFrac): Bias Hist (Test)")

    # ── 4.  Wilcoxon rank‑sum tests  (c1–c3 vs. c0) ─────────────────
    cat_test = cat_valid[test_idx]
    bias_by_cat = {c: y_pred_test[cat_test == c] - y_test[cat_test == c]
                   for c in range(4) if np.any(cat_test == c)}

    if 0 in bias_by_cat:
        for c in (1, 2, 3):
            if c in bias_by_cat:
                stat, p = ranksums(bias_by_cat[0], bias_by_cat[c])
                print(f"[NoBurnFrac] Wilcoxon c{c} vs c0: stat={stat:.3f}, p={p:.3g}")

    # ── 5.  Down‑sampling robustness check ───────────────────────────
    counts = {c: (cat_test == c).sum() for c in range(4) if (cat_test == c).any()}
    if counts:
        min_cat  = min(counts, key=counts.get)
        min_cnt  = counts[min_cat]
        print(f"[NoBurnFrac] Fewest test samples: c{min_cat} (N={min_cnt})")

        for c, n_c in counts.items():
            if n_c < min_cnt:  continue
            rmses, biases = [], []
            for _ in range(10):
                idx = npr.choice(np.where(cat_test == c)[0], size=min_cnt, replace=False)
                rmses.append(np.sqrt(mean_squared_error(y_test[idx], y_pred_test[idx])))
                biases.append((y_pred_test[idx] - y_test[idx]).mean())
            print(f"[NoBurnFrac] c{c}: mean RMSE={np.mean(rmses):.3f}, "
                  f"mean Bias={np.mean(biases):.3f}")

    # ── 6.  Save per‑cat obs / pred to .txt ──────────────────────────
    for c in range(4):
        sel = cat_test == c
        if not sel.any():  continue
        np.savetxt(f"/Users/yashnilmohanty/Desktop/obs_DOD_cat{c}_NoBF.txt",
                   y_test[sel],  fmt="%.6f")
        np.savetxt(f"/Users/yashnilmohanty/Desktop/pred_DOD_cat{c}_NoBF.txt",
                   y_pred_test[sel], fmt="%.6f")

    # ── 7.  Per‑cat scatter / hist ───────────────────────────────────
    for c in range(4):
        sel = cat_test == c
        if sel.any():
            plot_scatter(y_test[sel], y_pred_test[sel],
                         title=f"RF (NoBurnFrac): Test cat={c}")
            plot_bias_hist(y_test[sel], y_pred_test[sel],
                           title=f"RF (NoBurnFrac): Bias Hist cat={c}")

    # Color‑coded scatter for all test data
    plot_scatter_by_cat(y_test, y_pred_test, cat_test,
                        title="RF (NoBurnFrac): All Test Data by Cat")

    # ── 8.  **NEW** 10 % burn‑fraction‑bin metrics ───────────────────
    burn_flat  = ds["burn_fraction"].values.ravel(order="C")
    burn_valid = burn_flat[valid_mask]
    burn_test  = burn_valid[test_idx]

    ten_bins = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),
                (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),
                (0.8,0.9),(0.9,None)]
    print("\n=== 10 % burn‑fraction bins (NoBurnFrac, Test set) ===")
    evaluate_metrics_per_bin(y_test, y_pred_test, burn_test, ten_bins)

    # ── 9.  **NEW** simple bias map ──────────────────────────────────
    pix_full  = np.tile(np.arange(ds.dims["pixel"]), ds.dims["year"])
    pix_valid = pix_full[valid_mask]
    pix_test  = pix_valid[test_idx]
    produce_pixel_bias_map_simple(ds, pix_test, y_test, y_pred_test,
                                  title="Pixel Bias: Simple Grid (NoBurnFrac)")

    # ── 10.  Box‑plot by elevation / veg ‑ unchanged ────────────────
    elev = ds["Elevation"].values.ravel(order="C")[valid_mask][test_idx]
    veg  = ds["VegTyp"].values.ravel(order="C")[valid_mask][test_idx]
    plot_boxplot_dod_by_elev_veg(y_test, elev, veg,
                                 cat_label="AllTest (NoBurnFrac)")

    # ── 11.  Feature importance & top‑5 scatter ─────────────────────
    plot_top10_features(rf, feat_names,
                        title="RF (NoBurnFrac): Top‑10 Feature Importance")
    plot_top5_feature_scatter(rf, X_test, y_test, cat_test,
                              feat_names, title_prefix="(NoBurnFrac)")

    # ── 12.  **NEW** mean predicted / observed DoD maps ─────────────
    y_pred_all = rf.predict(X_valid)
    n_pix = ds.dims["pixel"]
    sum_p = np.zeros(n_pix);  sum_o = np.zeros(n_pix);  cnt = np.zeros(n_pix)
    for px, obs, pred in zip(pix_valid, y_valid, y_pred_all):
        sum_p[px] += pred;  sum_o[px] += obs;  cnt[px] += 1
    mean_pred = np.where(cnt > 0, sum_p / cnt, np.nan)
    mean_obs  = np.where(cnt > 0, sum_o / cnt, np.nan)

    plot_mean_dod_map_simple(ds, mean_pred,
                             title="Mean Predicted DoD (NoBurnFrac)")
    plot_mean_dod_map_simple(ds, mean_obs,
                             title="Mean Observed DoD (NoBurnFrac)")

    # ── 13.  Return fitted model ────────────────────────────────────
    return rf


############################################################
# MAIN
############################################################
if __name__=="__main__":
    # 1) read coords, compute pre-burn
    path_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc"
    pre_burn = compute_pre2004_burn(coords, path_pat, 2001, 2003)

    # 2) open final_dataset3, do cumsum => cat_2d
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset3.nc")
    cumsum_2d = compute_burn_cumsum_with_initial(ds, pre_burn)
    ds["burn_cumsum"] = (("year","pixel"), cumsum_2d)
    cat_2d = define_4cats_cumsum(cumsum_2d)

    # 3) flatten => explicitly excluding 'burn_fraction' & 'burn_cumsum'
    X_all, y_all, feat_names, valid_mask = flatten_spatiotemporal(ds, target_var="DOD")
    print("Feature names (No BurnFrac):", feat_names)

    # 4) run experiment => includes rank-sum, downsampling, saving .txt
    rf_model_noFrac = run_rf_excluding_burnfraction(
        X_all, y_all, cat_2d, valid_mask, ds, feat_names
    )
    print("DONE (NoBurnFrac).")
