#!/usr/bin/env python3
# ============================================================
#  Fire-ML  ·  LSTM experiment on final_dataset4.nc
#  70 % unburned-only training  →  evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
import time, xarray as xr, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ranksums, spearmanr
import numpy.random as npr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import socket, requests
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd

GLOBAL_VEGRANGE: np.ndarray = np.array([], dtype=int)

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  Spatial helpers (satellite background + per-pixel plotting)
# ────────────────────────────────────────────────────────────

def _satellite_available(timeout_s: int = 2) -> bool:
    url = ("https://services.arcgisonline.com/arcgis/rest/services/"
           "World_Imagery/MapServer")
    try:
        requests.head(url, timeout=timeout_s)
        return True
    except (requests.RequestException, socket.error):
        return False

USE_SAT = _satellite_available()

# one-time Web-Mercator extent of California
CA_LON_W, CA_LON_E = -125.0, -117.0
CA_LAT_S, CA_LAT_N =   37.0,   43.0
merc = ccrs.epsg(3857)
x0, y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())
x1, y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())
CA_EXTENT = [x0, y0, x1, y1]

STATES_SHP = "data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
STATES = gpd.read_file(STATES_SHP).to_crs(epsg=3857)

def _add_background(ax, zoom=6):
    """
    Add either satellite tiles (if available) or shaded relief + state borders.
    """
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    if USE_SAT:
        try:
            tiler = cimgt.GoogleTiles(style='satellite')
            tiler.request_timeout = 5
            ax.add_image(tiler, zoom, interpolation="nearest")
        except Exception:
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    "physical", "shaded_relief", "10m",
                    edgecolor="none", facecolor=cfeature.COLORS["land"]
                ),
                zorder=0
            )
    else:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "shaded_relief", "10m",
                edgecolor="none", facecolor=cfeature.COLORS["land"]
            ),
            zorder=0
        )
    STATES.boundary.plot(
        ax=ax,
        linewidth=0.6,
        edgecolor="black",
        zorder=2
    )

def dod_map_ca(ds, pix_idx, values, title,
               cmap="Blues", vmin=50, vmax=250):
    """
    Plot per-pixel DOD values over California with a common 50–250-day scale.
    """
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    merc = ccrs.epsg(3857)
    x, y = merc.transform_points(
        ccrs.Geodetic(), lon[pix_idx], lat[pix_idx]
    )[:, :2].T

    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6,5))
    _add_background(ax, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    sc = ax.scatter(
        x, y, c=values, cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=2, marker="s", transform=merc, zorder=3
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, label="DSD (days)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def bias_map_ca(ds, pix_idx, y_true, y_pred, title):
    """
    Plot clipped bias (Pred – Obs) per pixel, using a diverging scale:
    now flipped so positive = blue, negative = red.
    """
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    merc = ccrs.epsg(3857)
    x, y = merc.transform_points(
        ccrs.Geodetic(), lon[pix_idx], lat[pix_idx]
    )[:, :2].T

    bias = np.clip(y_pred - y_true, -60, 60)
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6,5))
    _add_background(ax, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    sc = ax.scatter(
        x, y, c=bias, cmap="seismic_r",
        norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60),
        s=2, marker="s", transform=merc, zorder=3
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Bias (Pred − Obs, days)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def boxplot_dod_by_elev_veg(y, elev, veg, tag):
    """
    Simply a side-by-side boxplot of raw DOD by (elev_bin, VegTyp).
    """
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin = np.digitize(elev, elev_edges) - 1
    uniq_veg = np.unique(veg)
    data, labels = [], []
    for ei in range(len(elev_edges)-1):
        for vv in uniq_veg:
            m = (elev_bin == ei) & (veg == vv)
            data.append(y[m])
            labels.append(f"E[{elev_edges[ei]}–{elev_edges[ei+1]}],V{vv}")
    plt.figure(figsize=(12,5))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(labels)+1), labels, rotation=90)
    plt.xlabel("(Elevation bin, VegTyp)")
    plt.ylabel("Raw DSD")
    plt.title(tag)
    plt.tight_layout()
    plt.show()

def heat_bias_by_elev_veg(y_true, y_pred, elev, veg, tag,
                          elev_edges=(500,1000,1500,2000,2500,3000,3500,4000,4500)):
    """
    Create an Elev×VegTyp grid of mean bias (Pred − Obs), clipped to ±60 days.
    Now uses GLOBAL_VEGRANGE to drop any veg columns that never occur.
    """
    bias      = y_pred - y_true
    elev_bin  = np.digitize(elev, elev_edges) - 1
    veg_range = GLOBAL_VEGRANGE
    n_veg     = len(veg_range)

    grid = np.full((len(elev_edges)-1, n_veg), np.nan)
    for ei in range(len(elev_edges)-1):
        for j, vv in enumerate(veg_range):
            sel = (elev_bin == ei) & (veg.astype(int) == vv)
            if sel.any():
                grid[ei, j] = np.nanmean(bias[sel])

    plt.figure(figsize=(8,4))
    im = plt.imshow(
        grid, cmap='seismic_r', vmin=-60, vmax=60,
        origin='lower', aspect='auto'
    )
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.gca().add_patch(
                plt.Rectangle((j-0.5, i-0.5), 1, 1,
                              ec='black', fc='none', lw=0.6)
            )
            if not np.isnan(grid[i, j]):
                plt.text(j, i, f"{grid[i, j]:.0f}",
                         ha='center', va='center', fontsize=6, color='k')

    plt.xticks(range(n_veg), [f"V{v}" for v in veg_range])
    plt.yticks(range(len(elev_edges)-1),
               [f"{elev_edges[i]}–{elev_edges[i+1]}" for i in range(len(elev_edges)-1)])
    plt.colorbar(im, label="Bias (days)")
    plt.title(tag)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────
#  plotting helpers  (scatter / bias-hist / feature plots)
# ────────────────────────────────────────────────────────────

def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn, mx = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label="1:1 line")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = (y_pred - y_true).mean();  r2 = r2_score(y_true, y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DSD");  plt.ylabel("Observed DSD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_bias_hist(y_true, y_pred, title, rng=(-100,300)):
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=50, range=rng, alpha=0.7)
    plt.axvline(res.mean(), color='k', ls='--', lw=2)
    plt.title(f"{title}\nMean={res.mean():.2f}, Std={res.std():.2f}")
    plt.xlabel("Bias (Pred-Obs)");  plt.ylabel("Count")
    plt.tight_layout();  plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title):
    plt.figure(figsize=(6,6))
    cols = {0:'red', 1:'yellow', 2:'green', 3:'blue'}
    for c,col in cols.items():
        m = cat==c
        if m.any():
            plt.scatter(y_pred[m], y_true[m], c=col, alpha=0.4, label=f"cat={c}")
    mn, mx = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = (y_pred - y_true).mean();  r2 = r2_score(y_true, y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DSD");  plt.ylabel("Observed DSD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_top10_perm_importance(imp, names, title):
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx])
    plt.xticks(range(10), [names[i] for i in idx], rotation=45, ha='right')
    plt.title(title);  plt.ylabel("Permutation importance (ΔMSE)")
    plt.tight_layout();  plt.show()

def plot_top5_feature_scatter(imp, X, y, cat, names, prefix):
    """
    Aggregated Top-5 feature scatter:
      • x-axis  = predictor value (mean of pixels sharing DoD & category)
      • y-axis  = observed DoD
      • horizontal bar = ±1 SD
      • one coloured line per cumulative-burn category
    """
    top5    = np.argsort(imp)[::-1][:5]
    colours = {0:'red', 1:'yellow', 2:'green', 3:'blue'}
    cats    = [0, 1, 2, 3]

    for f_idx in top5:
        fname = names[f_idx]
        x_all = X[:, f_idx]

        plt.figure(figsize=(7, 5))
        for c in cats:
            mask_c = (cat == c)
            if not mask_c.any():
                continue

            # Pearson r for legend (raw points, not aggregated)
            r_val = np.corrcoef(x_all[mask_c], y[mask_c])[0, 1]

            # aggregate by unique DoD
            dod_vals, mean_x, sd_x = [], [], []
            for d in np.unique(y[mask_c]):
                m_d = mask_c & (y == d)
                mean_x.append(np.mean(x_all[m_d]))
                sd_x.append(np.std(x_all[m_d]))
                dod_vals.append(d)
            dod_vals, mean_x, sd_x = map(np.asarray, (dod_vals, mean_x, sd_x))
            order = np.argsort(dod_vals)
            dod_vals, mean_x, sd_x = dod_vals[order], mean_x[order], sd_x[order]

            plt.errorbar(mean_x, dod_vals,
                         xerr=sd_x,
                         fmt='o', ms=4, lw=1,
                         color=colours[c], ecolor=colours[c],
                         alpha=0.8,
                         label=f"cat={c} (r={r_val:.2f})")
            plt.plot(mean_x, dod_vals, '-', color=colours[c], alpha=0.7)

        plt.xlabel(fname)
        plt.ylabel("Observed DSD")
        plt.title(f"{prefix}: {fname}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_top5_feature_scatter_binned(imp, X, y, cat, names, prefix, n_bins: int = 20):
    """
    Like plot_top5_feature_scatter, but bins each feature’s range into n_bins,
    and for each bin & category draws mean ±1 SD. Uses Spearman ρ + p-value in the legend.
    """
    top5 = np.argsort(imp)[::-1][:5]
    colours = {0: 'red', 1: 'yellow', 2: 'green', 3: 'blue'}
    cats = [0,1,2,3]

    for f_idx in top5:
        fname = names[f_idx]
        x_all = X[:, f_idx]

        # uniform bin edges + centres
        edges = np.linspace(x_all.min(), x_all.max(), n_bins+1)
        centres = 0.5*(edges[:-1] + edges[1:])

        plt.figure(figsize=(7,5))
        for c in cats:
            mask_c = (cat == c)
            if not mask_c.any():
                continue

            # Spearman rho + p-value
            rho, pval = spearmanr(x_all[mask_c], y[mask_c])

            y_mean, y_sd, x_valid = [], [], []
            for i in range(n_bins):
                m_bin = mask_c & (x_all >= edges[i]) & (x_all < edges[i+1])
                if not m_bin.any():
                    continue
                y_mean.append(y[m_bin].mean())
                y_sd.append(y[m_bin].std(ddof=0))
                x_valid.append(centres[i])

            if len(x_valid)==0:
                continue

            x_valid = np.array(x_valid)
            y_mean  = np.array(y_mean)
            y_sd    = np.array(y_sd)

            plt.errorbar(
                x_valid, y_mean,
                yerr=y_sd,
                fmt='o', ms=4, lw=1,
                color=colours[c], ecolor=colours[c],
                alpha=0.8,
                label=f"cat={c} (ρ={rho:.2f}, p={pval:.2g})"
            )
            plt.plot(x_valid, y_mean, '-', color=colours[c], alpha=0.7)

        plt.xlabel(fname)
        plt.ylabel("Observed DSD")
        plt.title(f"{prefix}: {fname}")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ────────────────────────────────────────────────────────────
#  feature-matrix helpers  (burn_fraction & burn_cumsum excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum','aorcsummerhumidity',
            'aorcsummerprecipitation','aorcsummerlongwave',
            'aorcsummershortwave','aorcsummertemperature'}
    ny = ds.sizes['year'];  feats = {}
    for v in ds.data_vars:
        if v.lower() in excl:  continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}:
            feats[v] = da.values
        elif set(da.dims)=={'pixel'}:
            feats[v] = np.tile(da.values, (ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok

# ────────────────────────────────────────────────────────────
#  permutation-importance for the (single-step) LSTM
# ────────────────────────────────────────────────────────────
def perm_importance_lstm(model, X_val_sc, y_val_sc, n_repeats=3, batch_size=512):
    N, nfeat = X_val_sc.shape
    base_pred = model.predict(X_val_sc.reshape(N,1,nfeat),
                              batch_size=batch_size).squeeze()
    base_mse = mean_squared_error(y_val_sc, base_pred)
    rng = np.random.RandomState(42)
    imps = np.zeros(nfeat)
    Xp = X_val_sc.copy()
    for f in range(nfeat):
        scores = []
        for _ in range(n_repeats):
            saved = Xp[:,f].copy()
            Xp[:,f] = rng.permutation(Xp[:,f])
            perm_pred = model.predict(Xp.reshape(N,1,nfeat),
                                      batch_size=batch_size).squeeze()
            scores.append(mean_squared_error(y_val_sc, perm_pred))
            Xp[:,f] = saved
        imps[f] = np.mean(scores) - base_mse
    return imps

# ────────────────────────────────────────────────────────────
#  main LSTM experiment (unburned-only training)
# ────────────────────────────────────────────────────────────
def lstm_unburned_experiment(X, y, cat2d, ok, ds, feat_names,
                             unburned_max_cat:int=0,
                             epochs=100, batch_size=512):
    thr = unburned_max_cat
    cat = cat2d.ravel(order='C')[ok]
    Xv, Yv = X[ok], y[ok]

    # unburned subset
    unb = cat <= thr
    log(f"  training on unburned (cat ≤ {thr}): N={unb.sum()}")

    # ── NEW: make a 70 % / 30 % split **inside every training category** ──
    train_idx, test_idx = [], []
    for c in range(thr + 1):                      # cats 0 … thr
        rows = np.where((cat == c) & unb)[0]
        if rows.size == 0:
            continue
        tr, te = train_test_split(rows,
                                  test_size=0.30,
                                  random_state=42)
        train_idx.append(tr)
        test_idx .append(te)

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    X_tr_raw, y_tr_raw = Xv[train_idx], Yv[train_idx]
    X_te_raw, y_te_raw = Xv[test_idx ], Yv[test_idx ]

    # scale inputs and output
    xsc, ysc = StandardScaler(), StandardScaler()
    X_tr_sc = xsc.fit_transform(X_tr_raw)
    y_tr_sc = ysc.fit_transform(y_tr_raw.reshape(-1,1)).ravel()
    X_te_sc = xsc.transform(X_te_raw)
    y_te_sc = ysc.transform(y_te_raw.reshape(-1,1)).ravel()
    X_all_sc= xsc.transform(Xv)

    # reshape to (N,1,n_feat)
    nfeat = X_tr_sc.shape[1]
    def to3D(a): return a.reshape(a.shape[0],1,nfeat)

    # build model
    model = Sequential()
    model.add(LSTM(32, input_shape=(1,nfeat)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(to3D(X_tr_sc), y_tr_sc,
              validation_data=(to3D(X_te_sc), y_te_sc),
              epochs=epochs, batch_size=batch_size, verbose=0)

    # helper to invert-scale predictions
    def inv(pred_sc): return ysc.inverse_transform(pred_sc.reshape(-1,1)).ravel()

    # A) unburned TRAIN / TEST plots
    yhat_tr = inv(model.predict(to3D(X_tr_sc), batch_size=batch_size).squeeze())
    plot_scatter(y_tr_raw, yhat_tr, f"LSTM TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr_raw, yhat_tr, f"Bias Hist: TRAIN (cat ≤ {thr})")

    yhat_te = inv(model.predict(to3D(X_te_sc), batch_size=batch_size).squeeze())
    plot_scatter(y_te_raw, yhat_te, f"LSTM TEST (cat ≤ {thr})")
    plot_bias_hist(y_te_raw, yhat_te, f"Bias Hist: TEST (cat ≤ {thr})")

    # B) all-data evaluation
    yhat_all = inv(model.predict(to3D(X_all_sc), batch_size=batch_size).squeeze())
    plot_scatter_by_cat(Yv, yhat_all, cat,
                        f"All data – colour by cat (thr={thr})")
    plot_bias_hist(Yv, yhat_all, f"Bias Hist: ALL data (thr={thr})")

    # ────────────────────────────────────────────────────────────
    #  spatial bias plots (overall + per category)
    # ────────────────────────────────────────────────────────────
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid = pix_full[ok]    # only rows where ok == True

    # overall pixel-bias map
    bias_map_ca(ds, pix_valid, Yv, yhat_all, f"Pixel Bias: ALL data (thr={thr})")

    # per-category pixel-bias maps & Elev×Veg bias grids
    for c in range(4):
        m = (cat == c)
        if not m.any():
            continue

        # per-category pixel-bias map
        bias_map_ca(
            ds,
            pix_valid[m],
            Yv[m],
            yhat_all[m],
            f"Pixel Bias – cat {c} (thr={thr})"
        )

        # Elevation & VegTyp vectors for this category
        elev_c = ds["Elevation"].values.ravel(order="C")[ok][m]
        veg_c  = ds["VegTyp"]   .values.ravel(order="C")[ok][m]

        # Elev×Veg bias grid for this category
        heat_bias_by_elev_veg(
            Yv[m],
            yhat_all[m],
            elev_c,
            veg_c,
            f"Elev×Veg Bias – cat {c} (thr={thr})"
        )

    # C) per-category plots + Wilcoxon tests
    bias = yhat_all - Yv
    bias_by_cat = {c:bias[cat==c] for c in range(4) if (cat==c).any()}
    for c in range(4):
        m = cat == c
        if m.any():
            plot_scatter(Yv[m], yhat_all[m], f"Category {c} (thr={thr})")
            plot_bias_hist(Yv[m], yhat_all[m],
                           f"Bias Hist: cat={c} (thr={thr})")
    if 0 in bias_by_cat:
        print("\nWilcoxon rank-sum (bias difference vs cat 0)")
        for c in (1,2,3):
            if c in bias_by_cat:
                s,p = ranksums(bias_by_cat[0], bias_by_cat[c])
                print(f"  cat {c} vs 0 → stat={s:.3f}, p={p:.3g}")

    # D) permutation-based feature importance & Top-5 plots
    imp = perm_importance_lstm(model, X_te_sc, y_te_sc,
                               n_repeats=3, batch_size=batch_size)
    plot_top10_perm_importance(imp, feat_names,
                               f"Top-10 Permutation Importance (thr={thr})")
    plot_top5_feature_scatter(imp, Xv, Yv, cat, feat_names,
                              f"Top-5 (thr={thr})")
    plot_top5_feature_scatter_binned(
        imp,           # permutation-importance array
        Xv,            # all features
        Yv,            # observed DOD
        cat,           # category array
        feat_names,    # feature names
        f"Top-5 (thr={thr})"
    )

    return model

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__=="__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

    # cumulative-burn categories (stored as burn_cumsum)
    bc = ds["burn_cumsum"].values
    cat_2d = np.zeros_like(bc, dtype=int)
    cat_2d[bc < 0.25]                  = 0
    cat_2d[(bc >= 0.25) & (bc < 0.50)] = 1
    cat_2d[(bc >= 0.50) & (bc < 0.75)] = 2
    cat_2d[bc >= 0.75]                 = 3
    log("categories (c0-c3) computed")

    # feature matrix (burn_fraction excluded)
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    log(f"feature matrix ready – {ok.sum()} valid samples, {len(feat_names)} predictors")

    # build global VegTyp list for bias grids
    veg_all = ds["VegTyp"].values.ravel(order="C")[ok].astype(int)
    GLOBAL_VEGRANGE = np.unique(veg_all)

    # ── RUN #1  (unburned = cat 0) ─────────────────────────────
    log("\n=== LSTM RUN #1 : unburned = cat 0 (cumsum < 0.25) ===")
    lstm_run1 = lstm_unburned_experiment(
        X_all, y_all, cat_2d, ok, ds, feat_names,
        unburned_max_cat=0, epochs=100, batch_size=512
    )

    # ── RUN #2  (unburned = cat 0 + 1) ─────────────────────────
    log("\n=== LSTM RUN #2 : unburned = cat 0 + 1 (cumsum < 0.50) ===")
    lstm_run2 = lstm_unburned_experiment(
        X_all, y_all, cat_2d, ok, ds, feat_names,
        unburned_max_cat=1, epochs=100, batch_size=512
    )

    log("ALL DONE.")
