#!/usr/bin/env python3
# ============================================================
#  Fire‑ML experiment on final_dataset4.nc
#  70 % unburned‑only training → evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
# ─── core libs ───────────────────────────────────────────────
import time, socket, requests
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ranksums
import xarray as xr

# ─── scikit-learn ────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── geo / plotting ──────────────────────────────────────────
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import geopandas as gpd
import xyzservices.providers as xyz
import cartopy.io.img_tiles as cimgt
from pathlib import Path

# typing
from typing import Dict, List, Tuple, Optional

PIX_SZ = 2
STATES_SHP = "data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
STATES = gpd.read_file(STATES_SHP).to_crs(epsg=3857)
# ─── California lon/lat rectangle  (PlateCarree) ─────────────
CA_LON_W, CA_LON_E = -125.0, -117.0   # west, east
CA_LAT_S, CA_LAT_N =   37.0,   43.0   # south, north

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  generic plotting helpers
# ────────────────────────────────────────────────────────────
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--',label="1:1 line")
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean();  r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD");  plt.ylabel("Observed DoD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_bias_hist(y_true, y_pred, title, rng=(-100,300)):
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=50, range=rng, alpha=0.7)
    plt.axvline(res.mean(), color='k', ls='--', lw=2)
    plt.title(f"{title}\nMean={res.mean():.2f}, Std={res.std():.2f}")
    plt.xlabel("Bias (Pred‑Obs)");  plt.ylabel("Count")
    plt.tight_layout();  plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title):
    plt.figure(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    for c,col in cols.items():
        m = cat==c
        if m.any():
            plt.scatter(y_pred[m], y_true[m], c=col, alpha=0.4, label=f"cat={c}")
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean();  r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD");  plt.ylabel("Observed DoD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_top10_features(rf, names, title):
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx])
    plt.xticks(range(10), [names[i] for i in idx], rotation=45, ha='right')
    plt.title(title);  plt.ylabel("Feature importance")
    plt.tight_layout();  plt.show()

# ────────────────────────────────────────────────────────────
#  NEW aggregated Top‑5 feature‑scatter (mean ± 1 SD per DoD)
# ────────────────────────────────────────────────────────────
def plot_top5_feature_scatter(rf, X, y, cat, names, prefix):
    """
    • x‑axis = predictor value, y‑axis = observed DoD  
    • for every unique DoD and every category c0..c3:  
        – compute mean(x) & std(x) among pixels with that DoD+cat  
        – plot a single point (mean, DoD) with a horizontal ±1 SD bar  
    • connect the points of each category with a line  
    • legend shows Pearson r for **each category** (computed on *all* raw
      points of that category, not the aggregated means)  
    Colours: red(c0) / yellow(c1) / green(c2) / blue(c3)
    """
    imp = rf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    colours = {0:'red', 1:'yellow', 2:'green', 3:'blue'}
    cats    = [0,1,2,3]

    for f_idx in top5:
        fname = names[f_idx]
        x_all = X[:,f_idx]

        plt.figure(figsize=(7,5))
        for c in cats:
            mask_c = (cat == c)
            if not mask_c.any():          # skip empty category
                continue

            # Pearson r for legend (all raw points of this category)
            r_val = np.corrcoef(x_all[mask_c], y[mask_c])[0,1]

            # aggregate by unique DoD
            dod_vals, mean_x, sd_x = [], [], []
            for d in np.unique(y[mask_c]):
                m_d = mask_c & (y == d)
                mean_x.append( np.mean(x_all[m_d]) )
                sd_x  .append( np.std (x_all[m_d]) )
                dod_vals.append(d)
            dod_vals = np.array(dod_vals)
            mean_x   = np.array(mean_x)
            sd_x     = np.array(sd_x)

            # sort by DoD to get a nice line
            order = np.argsort(dod_vals)
            dod_vals = dod_vals[order]
            mean_x   = mean_x[order]
            sd_x     = sd_x[order]

            # error bars (horizontal) + line
            plt.errorbar(mean_x, dod_vals,
                         xerr=sd_x,
                         fmt='o', ms=4, lw=1,
                         color=colours[c],
                         ecolor=colours[c],
                         alpha=0.8,
                         label=f"cat={c} (r={r_val:.2f})")
            plt.plot(mean_x, dod_vals, '-', color=colours[c], alpha=0.7)

        plt.xlabel(fname)
        plt.ylabel("Observed DoD")
        plt.title(f"{prefix}: {fname}")
        plt.legend()
        plt.tight_layout();  plt.show()

# (the rest of the script – spatial helpers, feature‑matrix builders,
#  eval‑bins, rf_unburned_experiment, main() – is IDENTICAL to the
#  previous version and therefore omitted here for brevity.  Simply
#  replace the old `plot_top5_feature_scatter()` with the new one
#  above, keep everything else unchanged.)

# ------------------------------------------------------------
# NEW  :  Top-5 feature scatter – 20 feature bins on the x-axis
# ------------------------------------------------------------
def plot_top5_feature_scatter_binned(
        rf: RandomForestRegressor,
        X: np.ndarray,
        y: np.ndarray,
        cat: np.ndarray,
        names: List[str],
        prefix: str,
        n_bins: int = 20):
    """
    Same colour/legend convention as the original scatter.
    • divide the feature range into *n_bins* equal-width bins
    • for every bin & every category:
        – x = bin centre
        – y = mean(DOD) of points in that bin & category
        – vertical bar = ±1 SD(DOD)
    • connect the 20 points of each category with a line.
    """
    imp  = rf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    colours = {0: 'red', 1: 'yellow', 2: 'green', 3: 'blue'}
    cats    = [0, 1, 2, 3]

    for f_idx in top5:
        fname = names[f_idx]
        x_all = X[:, f_idx]

        # fixed bin edges & centres (equal width)
        edges   = np.linspace(x_all.min(), x_all.max(), n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])

        plt.figure(figsize=(7, 5))
        for c in cats:
            mask_c = (cat == c)
            if not mask_c.any():
                continue

            # corr for the legend
            r_val = np.corrcoef(x_all[mask_c], y[mask_c])[0, 1]

            y_mean, y_sd, x_valid = [], [], []
            for i in range(n_bins):
                m_bin = mask_c & (x_all >= edges[i]) & (x_all < edges[i + 1])
                if not m_bin.any():
                    continue
                y_mean.append(y[m_bin].mean())
                y_sd  .append(y[m_bin].std(ddof=0))
                x_valid.append(centres[i])

            if not x_valid:    # nothing fell into any bin
                continue

            y_mean = np.array(y_mean)
            y_sd   = np.array(y_sd)
            x_valid= np.array(x_valid)

            plt.errorbar(x_valid, y_mean,
                         yerr=y_sd,
                         fmt='o', ms=4, lw=1,
                         color=colours[c], ecolor=colours[c],
                         alpha=0.8,
                         label=f"cat={c} (r={r_val:.2f})")
            plt.plot(x_valid, y_mean, '-', color=colours[c], alpha=0.7)

        plt.xlabel(fname)
        plt.ylabel("Observed DoD")
        plt.title(f"{prefix} (binned): {fname}")
        plt.legend()
        plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
# EXTRA DIAGNOSTIC PLOTS  (box-plots & histograms)
# ------------------------------------------------------------------
def boxplot_dod_by_cat(y_obs, y_pred, cat, title_prefix, fname_base=None):
    """Two side-by-side box-plots: observed & predicted DOD per category."""
    cats = [0, 1, 2, 3]
    data_obs = [y_obs[cat == c] for c in cats]
    data_pred = [y_pred[cat == c] for c in cats]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].boxplot(data_obs, showmeans=True)
    axs[0].set_title(f"{title_prefix} – OBSERVED")
    axs[1].boxplot(data_pred, showmeans=True)
    axs[1].set_title(f"{title_prefix} – PREDICTED")
    for ax in axs:
        ax.set_xticklabels([f"c{c}" for c in cats]);  ax.set_xlabel("Category")
    axs[0].set_ylabel("DoD (days)")
    fig.tight_layout()
    if fname_base:
        fig.savefig(f"{fname_base}.png", dpi=300)
    plt.show()


def boxplot_top5_predictors(X, feat_names, cat, rf, prefix, fname_base=None):
    """For each of the RF top-5 features make a one-panel box-plot per cat."""
    idx = np.argsort(rf.feature_importances_)[::-1][:5]
    cats = [0, 1, 2, 3]
    for i in idx:
        data = [X[cat == c, i] for c in cats]
        plt.figure(figsize=(5, 3.5))
        plt.boxplot(data, showmeans=True)
        plt.xticks(range(1, 5), [f"c{c}" for c in cats])
        plt.ylabel(feat_names[i])
        plt.title(f"{prefix}: {feat_names[i]}")
        plt.tight_layout()
        if fname_base:
            plt.savefig(f"{fname_base}_{feat_names[i]}.png", dpi=300)
        plt.show()


def transparent_histogram_by_cat(values, cat, title, alpha=0.35,
                                 colors={0:'red',1:'yellow',2:'green',3:'blue'},
                                 fname=None):
    """Overlaid semi-transparent histograms by category (same bin range)."""
    plt.figure(figsize=(6,4))
    rng = (np.nanmin(values), np.nanmax(values))
    bins = 40
    for c, col in colors.items():
        sel = cat == c
        if sel.any():
            plt.hist(values[sel], bins=bins, range=rng,
                     alpha=alpha, color=col, label=f"c{c}", density=True)
    plt.xlabel("DoD (days)")
    plt.ylabel("relative freq.")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()

# ── refined spatial helpers ─────────────────────────────────────────
def _setup_ca_axes(title:str):
    ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125,-113,32,42], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, lw=0.6)
    ax.coastlines(resolution="10m", lw=0.5)
    ax.set_title(title)
    return ax

TILER = cimgt.GoogleTiles(style='satellite')   # or StamenTerrain
TILER.request_timeout = 5

# ------------------------------------------------------------
# helper: tight Mercator extent around all finite pixels
# ------------------------------------------------------------
def dataset_extent_mercator(ds, pad_km=100):        # 100 km is plenty
    """
    Return [xmin, ymin, xmax, ymax] in EPSG:3857 that encloses every finite
    (lon,lat) pair in *ds*, plus pad_km on all sides.  Longitudes that are
    stored in 0…360° are wrapped to −180…180° first.
    """
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()

    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        raise RuntimeError("no finite lat/lon values!")

    lon = lon.copy()
    lon[lon > 180] -= 360                    # wrap 0–360° → −180…180°

    merc = ccrs.epsg(3857)
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[finite], lat[finite])[:, :2].T

    pad = pad_km * 1_000                     # km → m
    # clamp to the legal Web-Mercator range just in case
    max_wm = 20_037_508.342789244
    x_min = max(-max_wm, x.min() - pad)
    x_max = min( max_wm, x.max() + pad)
    y_min = max(-max_wm, y.min() - pad)
    y_max = min( max_wm, y.max() + pad)
    return [x_min, y_min, x_max, y_max]

# call *once* after you open the dataset
CA_EXTENT = None

def _satellite_available(timeout_s: int = 2) -> bool:
    """Quick HEAD request; returns True if Esri tiles reachable."""
    url = "https://services.arcgisonline.com/arcgis/rest/services" \
          "/World_Imagery/MapServer"
    try:
        requests.head(url, timeout=timeout_s)        # no auth, very small
        return True
    except (requests.RequestException, socket.error):
        return False

USE_SAT   = _satellite_available()
print("[INFO] satellite imagery available:", USE_SAT)

# ------------------------------------------------------------------
# background painter  (unchanged except that it uses CA_EXTENT)
# ------------------------------------------------------------------

_RELIEF = cfeature.NaturalEarthFeature(
            "physical", "shaded_relief", "10m",
            edgecolor="none", facecolor=cfeature.COLORS["land"])

# --- keep this helper; we still need the extent -----------------
def add_background(ax, extent_merc, zoom=6):
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
              crs=ccrs.PlateCarree())
    try:
        ax.add_image(TILER, zoom, interpolation="nearest")
    except Exception as e:
        print("⚠︎ satellite tiles skipped:", e)
        ax.add_feature(_RELIEF, zorder=0)   # shaded relief fallback
    STATES.boundary.plot(ax=ax, linewidth=.6, edgecolor="black", zorder=2)


# ------------------------------------------------------------------
# DoD and Bias maps   (background FIRST, pixels ON TOP)
# ------------------------------------------------------------------
def dod_map_ca(ds, pix_idx, values, title,
               cmap="Blues", vmin=50, vmax=250):
    merc = ccrs.epsg(3857)
    lat, lon = ds["latitude"].values.ravel(), ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    add_background(ax, CA_EXTENT)                # background tiles
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())        # <- NEW, hard clip
    sc = ax.scatter(x, y, c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=PIX_SZ, marker="s", transform=merc, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=.8, label="DoD (days)")
    ax.set_title(title); plt.tight_layout(); plt.show()

def bias_map_ca(ds, pix_idx, y_true, y_pred, title):
    merc = ccrs.epsg(3857)
    lat, lon = ds["latitude"].values.ravel(), ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    bias = np.clip(y_pred - y_true, -60, 60)
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    add_background(ax, CA_EXTENT)                           # ← changed
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())        # <- NEW, hard clip
    sc = ax.scatter(x, y, c=bias, cmap="seismic",
                    norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60),
                    s=PIX_SZ, marker="s", transform=merc, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=.8, label="Bias (Pred-Obs, days)")
    ax.set_title(title); plt.tight_layout(); plt.show()



def boxplot_dod_by_elev_veg(y, elev, veg, tag):
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin = np.digitize(elev, elev_edges) - 1
    uniq_veg = np.unique(veg)
    data, labels = [], []
    for ei in range(len(elev_edges) - 1):
        for vv in uniq_veg:
            m = (elev_bin == ei) & (veg == vv)
            data.append(y[m])
            labels.append(f"E[{elev_edges[ei]}‑{elev_edges[ei+1]}],V{vv}")
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
    plt.xlabel("(Elevation bin, VegTyp)")
    plt.ylabel("Raw DoD")
    plt.title(tag)
    plt.tight_layout()
    plt.show()

def heat_bias_by_elev_veg(y_true, y_pred, elev, veg, tag,
                          elev_edges=(500,1000,1500,2000,2500,3000,3500,4000,4500)):

    bias       = y_pred - y_true
    elev_bin   = np.digitize(elev, elev_edges) - 1

    # ---- fixed VegTyp columns 1 … 23 ----------------------------------
    veg_range  = np.arange(1, 24)                # 23 columns
    n_veg      = len(veg_range)

    grid = np.full((len(elev_edges)-1, n_veg), np.nan)

    for ei in range(len(elev_edges)-1):
        for vv in veg_range:
            m = (elev_bin == ei) & (veg.astype(int) == vv)
            if m.any():
                grid[ei, vv-1] = np.nanmean(bias[m])   # vv-1 because 0-index

    plt.figure(figsize=(8,4))
    im = plt.imshow(grid, cmap='seismic', vmin=-60, vmax=60,
                    origin='lower', aspect='auto')

    # --- black border around every cell ---------------------------
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.gca().add_patch(
                plt.Rectangle((j-0.5, i-0.5), 1, 1,
                              ec='black', fc='none', lw=.6)
            )
            # value label
            if not np.isnan(grid[i, j]):
                plt.text(j, i, f"{grid[i, j]:.0f}",
                         ha='center', va='center', fontsize=7, color='k')

    plt.xticks(range(n_veg), [f"V{v}" for v in veg_range])
    plt.yticks(range(len(elev_edges)-1),
               [f"{elev_edges[i]}–{elev_edges[i+1]}" for i in range(len(elev_edges)-1)])
    plt.colorbar(im, label="Bias (days)")
    plt.title(tag)
    plt.tight_layout();  plt.show()



# ────────────────────────────────────────────────────────────
#  helpers to build feature matrix (burn_fraction excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum','aorcsummerhumidity',
            'aorcsummerprecipitation','aorcsummerlongwave',
            'aorcsummershortwave','aorcsummertemperature'}
    ny = ds.sizes["year"]
    feats = {}
    for v in ds.data_vars:
        if v.lower() in excl:
            continue
        da = ds[v]
        if set(da.dims) == {"year", "pixel"}:
            feats[v] = da.values
        elif set(da.dims) == {"pixel"}:
            feats[v] = np.tile(da.values, (ny, 1))
    return feats


def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok


# ────────────────────────────────────────────────────────────
#  evaluation helper: metrics in 10 % burn‑fraction bins
# ────────────────────────────────────────────────────────────
def eval_bins(y, yp, burn, bins):
    for lo, hi in bins:
        sel = (burn > lo) if hi is None else ((burn >= lo) & (burn < hi))
        tag = f">{lo*100:.0f}%" if hi is None else f"{lo*100:.0f}-{hi*100:.0f}%"
        if sel.sum() == 0:
            print(f"{tag}: N=0")
            continue
        rmse = np.sqrt(mean_squared_error(y[sel], yp[sel]))
        bias = (yp[sel] - y[sel]).mean()
        r2 = r2_score(y[sel], yp[sel])
        print(
            f"{tag}: N={sel.sum():4d}  RMSE={rmse:6.2f}  "
            f"Bias={bias:7.2f}  R²={r2:6.3f}"
        )


# ────────────────────────────────────────────────────────────
#  main RF routine  (unburned‑only training)
# ────────────────────────────────────────────────────────────
def rf_unburned_experiment(
    X,
    y,
    cat2d,
    ok,
    ds,
    feat_names,
    unburned_max_cat: int = 0,
):
    """
    • train on samples with cat ≤ unburned_max_cat
    • 70/30 split inside that subset
    • evaluate everywhere + all requested plots / stats
    """
    thr = unburned_max_cat
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]

    # ──────── Changed Part ────────

    # ── NEW: make a 70 % / 30 % split **inside every training category** ──
    train_idx, test_idx = [], []
    for c in range(thr + 1):                 # categories 0 … thr   (thr = 0 or 1)
        rows = np.where(cat == c)[0]         # only unburned cats
        if rows.size == 0:                   # just in case
            continue
        tr, te = train_test_split(rows,
                                test_size = 0.30,
                                random_state = 42)
        
        # print the size of the datasets
        print(tr.size)
        print(te.size)
        
        train_idx.append(tr)
        test_idx .append(te)

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)
    cat_te = cat[test_idx]

    X_tr, y_tr = Xv[train_idx], Yv[train_idx]      # **train** = 70 % of allowed cats
    X_te, y_te = Xv[test_idx ], Yv[test_idx ]      # **internal test** = 30 % of same


    # ──────── Changed Part ────────

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # ── A. unburned train / test plots ────────────────────────────
    plot_scatter(y_tr, rf.predict(X_tr), f"Unburned TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr, rf.predict(X_tr), f"Bias Hist: Unburned TRAIN (cat ≤ {thr})")

    y_hat_te = rf.predict(X_te)

    # --- NEW test-set box-plots & histograms -------------------------------
    boxplot_dod_by_cat(y_te, y_hat_te, cat_te,
                    title_prefix="TEST 30 %")

    transparent_histogram_by_cat(y_te,     cat_te,
                                "Observed DoD – TEST 30 %",
                                fname=None)
    transparent_histogram_by_cat(y_hat_te, cat_te,
                                "Predicted DoD – TEST 30 %",
                                fname=None)

    # box-plots for RF top-5 predictors (full sample, but you could
    # do the same on X_te by passing X_te instead of Xv)
    boxplot_top5_predictors(Xv, feat_names, cat,
                            rf, prefix="Top-5 predictors")


    plot_scatter(y_te, y_hat_te, f"Unburned TEST (cat ≤ {thr})")
    plot_bias_hist(y_te, y_hat_te, f"Bias Hist: Unburned TEST (cat ≤ {thr})")

    # ── B. evaluate on all valid samples ──────────────────────
    y_hat_all = rf.predict(Xv)

    # --- NEW global box-plots & histograms ---------------------------------
    boxplot_dod_by_cat(Yv, y_hat_all, cat,
                    title_prefix="FULL SAMPLE")

    # transparent full-sample observed / predicted histograms
    transparent_histogram_by_cat(Yv,        cat,
                                "Observed DoD – FULL sample",
                                fname=None)
    transparent_histogram_by_cat(y_hat_all, cat,
                                "Predicted DoD – FULL sample",
                                fname=None)


    plot_scatter_by_cat(
        Yv, y_hat_all, cat, f"All data – colour by cat (thr={thr})"
    )
    plot_bias_hist(Yv, y_hat_all, f"Bias Hist: ALL data (thr={thr})")

    # pixel‑level bias map
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    bias_map_ca(
        ds,
        pix_full[ok],
        Yv,
        y_hat_all,
        f"Pixel Bias: ALL data (thr={thr})",
    )

    # ── C. per‑cat scatter / hist, Wilcoxon tests, DoD maps, Bias maps ──
    bias_all = y_hat_all - Yv
    bias_by_cat = {c: bias_all[cat == c] for c in range(4) if (cat == c).any()}

    pix_valid = pix_full[ok]  # pixel index per row of Xv/Yv

    for c in range(4):
        m = cat == c
        if not m.any():
            continue

        # scatter / hist
        plot_scatter(
            Yv[m], y_hat_all[m], f"Category {c} (thr={thr})"
        )
        plot_bias_hist(
            Yv[m], y_hat_all[m], f"Bias Hist: cat={c} (thr={thr})"
        )

        # observed & predicted DoD maps
        dod_map_ca(
            ds,
            pix_valid[m],
            Yv[m],
            f"Observed DoD – cat {c} (thr={thr})",
            cmap="Blues",
        )
        dod_map_ca(
            ds,
            pix_valid[m],
            y_hat_all[m],
            f"Predicted DoD – cat {c} (thr={thr})",
            cmap="Blues",
        )

        # per‑category pixel‑bias map
        bias_map_ca(
            ds,
            pix_valid[m],
            Yv[m],
            y_hat_all[m],
            f"Pixel Bias – cat {c} (thr={thr})",
        )

        # per‑category Elev×Veg box‑plot
        elev = ds["Elevation"].values.ravel(order="C")[ok][m]
        veg = ds["VegTyp"].values.ravel(order="C")[ok][m]
        boxplot_dod_by_elev_veg(
            Yv[m], elev, veg, f"cat={c}, thr={thr}"
        )

        heat_bias_by_elev_veg(Yv[m], y_hat_all[m], elev, veg,
                      f"Elev×Veg Bias – cat {c} (thr={thr})")

    if 0 in bias_by_cat:
        print("\nWilcoxon rank‑sum (bias difference vs cat 0)")
        for c in (1, 2, 3):
            if c in bias_by_cat:
                s, p = ranksums(bias_by_cat[0], bias_by_cat[c])
                print(f"  cat {c} vs 0 → stat={s:.3f}, p={p:.3g}")

    # ── D. 10 % burn‑fraction bins (eval‑only) ────────────────────
    bf = ds["burn_fraction"].values.ravel(order="C")[ok]
    bins10 = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, None),
    ]
    print("\nPerformance by 10 % burn‑fraction bins:")
    eval_bins(Yv, y_hat_all, bf, bins10)

    # ── E. down‑sampling robustness  (k=100 runs) ─────────────────
    counts = {c:(cat==c).sum() for c in range(4) if (cat==c).any()}
    k = min(counts.values())                     # smallest class size
    ref_cat = min(counts, key=counts.get)        # class with size k
    print(f"\nDown‑sampling robustness: k={k}, runs=100  (ref = cat{ref_cat})")

    metrics: Dict[int, Dict[str, List[float]]] = {c:{'bias':[], 'rmse':[], 'r2':[]}
                                                  for c in counts}

    for c, n in counts.items():
        idx_c = np.where(cat==c)[0]
        for _ in range(100):
            sub = npr.choice(idx_c, size=k, replace=False)
            y_s   = Yv[sub]
            yhat  = y_hat_all[sub]
            metrics[c]['bias'].append((yhat - y_s).mean())
            metrics[c]['rmse'].append(np.sqrt(mean_squared_error(y_s, yhat)))
            metrics[c]['r2'  ].append(r2_score(y_s, yhat))

    # ── merged histogram figure ───────────────────────────────
    
    orig_stats = {}
    for c in counts:
        m = cat == c
        orig_stats[c] = {
            'bias': (y_hat_all[m] - Yv[m]).mean(),
            'rmse': np.sqrt(mean_squared_error(Yv[m], y_hat_all[m])),
            'r2'  : r2_score(Yv[m], y_hat_all[m])
        }

    # colours for every category
    col_distr = {0: 'black', 1: 'blue', 3: 'red'}   # cats shown as histograms
    col_line  = {2: 'grey'}                         # cats shown *only* as a line

    fig = plt.figure(figsize=(15, 4))
    for j, (key, lab) in enumerate([('bias', 'Mean Bias'),
                                    ('rmse', 'RMSE'),
                                    ('r2',   'R²')], 1):
        ax = fig.add_subplot(1, 3, j)

        # 1) histograms + their dashed means (cats 0,1,3)
        for c, col in col_distr.items():
            ax.hist(metrics[c][key],
                    bins=10, alpha=.45, color=col, label=f"cat{c}")
            ax.axvline(np.mean(metrics[c][key]), color=col, ls='--', lw=2)   # dashed

            # solid vertical line = metric on *all* samples of that cat
            ax.axvline(orig_stats[c][key], color=col, ls='-', lw=2)

        # 2) line-only categories (just cat2 here)
        for c, col in col_line.items():
            # mean of robustness runs  (dashed)   ← ADD a label here
            ax.axvline(np.mean(metrics[c][key]),
                    color=col, ls='--', lw=2, label=f"cat{c}")

            # mean of original full-sample metric  (solid) – no label to avoid legend dupes
            ax.axvline(orig_stats[c][key], color=col, ls='-', lw=2)

        ax.set_xlabel(lab)
        ax.set_title(lab)
        if j == 1:                     # only put the legend on the middle panel
            ax.legend()
    fig.suptitle(f"Down-sampling distributions (k={k})")
    fig.tight_layout()
    plt.show()

    # ── F. feature importance & top‑5 scatter ─────────────────────
    # --- F. feature importance & both Top-5 scatter variants --------
    plot_top10_features(rf, feat_names,
                        f"Top-10 Feature Importance (thr={unburned_max_cat})")
    plot_top5_feature_scatter(       rf, Xv, Yv, cat, feat_names,
                        f"Top-5 thr={unburned_max_cat}")
    plot_top5_feature_scatter_binned(rf, Xv, Yv, cat, feat_names,
                        f"Top-5 thr={unburned_max_cat}")

    return rf


# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")
    # ---- fixed mercator extent that encloses the CA rectangle ----
    merc   = ccrs.epsg(3857)
    x0, y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())
    x1, y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())
    CA_EXTENT = [x0, y0, x1, y1]          # will be used by add_background()
    print("[DEBUG] CA extent:", CA_EXTENT)      # should be ~ −1.4e7 … −1.25e7 etc.

    # cumulative‑burn categories already stored as burn_cumsum
    bc = ds["burn_cumsum"].values  # (year, pixel)
    cat_2d = np.zeros_like(bc, dtype=int)
    cat_2d[bc < 0.25] = 0
    cat_2d[(bc >= 0.25) & (bc < 0.50)] = 1
    cat_2d[(bc >= 0.50) & (bc < 0.75)] = 2
    cat_2d[bc >= 0.75] = 3
    log("categories (c0‑c3) computed")

    # build feature matrix (burn_fraction excluded)
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    log(
        f"feature matrix ready – {ok.sum()} valid samples, "
        f"{len(feat_names)} predictors"
    )

    # ── run #1 : unburned = cat 0 only  ───────────────────────────
    log("\n=== RUN #1 : unburned = cat 0 (cumsum < 0.25) ===")
    rf_run1 = rf_unburned_experiment(
        X_all, y_all, cat_2d, ok, ds, feat_names, unburned_max_cat=0
    )

    # ── run #2 : unburned = cat 0 + 1  ───────────────────────────
    log("\n=== RUN #2 : unburned = cat 0 + 1 (cumsum < 0.50) ===")
    rf_run2 = rf_unburned_experiment(
        X_all, y_all, cat_2d, ok, ds, feat_names, unburned_max_cat=1
    )

    log("ALL DONE.")
