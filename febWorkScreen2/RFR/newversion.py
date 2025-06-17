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
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde
import xarray as xr

# ─── scikit-learn ────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

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

PIX_SZ = 0.5
STATES_SHP = "data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
STATES = gpd.read_file(STATES_SHP).to_crs(epsg=3857)
# ─── California lon/lat rectangle  (PlateCarree) ─────────────
CA_LON_W, CA_LON_E = -124.5, -117.5   # west, east
CA_LAT_S, CA_LAT_N =   37,   42.5   # south, north
GLOBAL_VEGRANGE: np.ndarray = np.array([], dtype=int)
FONT_LABEL  = 14   # x/y axis labels
FONT_TICK   = 12   # tick labels
FONT_LEGEND = 12   # legend text

# add at top of script:
VEG_NAMES = {
    1:"Urban/Built-Up",    2:"Dry Cropland/Pasture",
    3:"Irrigated Crop/Pasture",  4:"Mixed Dry/Irrig.", 
    5:"Crop/Grass Mosaic", 6:"Crop/Wood Mosaic",
    7:"Grassland", 8:"Shrubland", 9:"Mixed Shrub/Grass",
   10:"Savanna", 11:"Deciduous Broadleaf",12:"Deciduous Needleleaf",
   13:"Evergreen Broadleaf",14:"Evergreen Needleleaf",15:"Mixed Forest",
   16:"Water",17:"Herb. Wetland",18:"Wooded Wetland",19:"Barren",
   20:"Herb. Tundra",21:"Wooded Tundra",22:"Mixed Tundra",
   23:"Bare Ground Tundra",24:"Snow/Ice",25:"Playa",26:"Lava",27:"White Sand"
}

NICE_NAME = {}
for season in ("Fall","Winter","Spring","Summer"):
    for feat in ("Temperature","Precipitation","Humidity","Shortwave","Longwave"):
        key = f"aorc{season}{feat}"
        arrow = "↓" if feat=="Shortwave" else ""
        NICE_NAME[key] = f"{season} {feat}{arrow}"
NICE_NAME["peakValue"] = "Peak SWE"
NICE_NAME["Elevation"]  = "Elevation (m)"
NICE_NAME["slope"]      = "Slope"
NICE_NAME["aspect_ratio"] = "Aspect Ratio"
NICE_NAME["VegTyp"] = "Vegetation Type"
NICE_NAME["sweWinter"] = "Winter SWE"
NICE_NAME["burn_fraction"] = "Burn Fraction"

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  generic plotting helpers
# ────────────────────────────────────────────────────────────
def plot_scatter(y_true, y_pred, title=None):
    """
    Simple scatter with 1:1 line, tight square axes with 5% padding.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_pred, y_true, alpha=0.3)
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred - y_true).mean()
        r2   = r2_score(y_true, y_pred)
        ax.set_title(
            f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.2f}",
            fontsize=FONT_LABEL
        )
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD",  fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def plot_density_scatter_by_cat(y_true, y_pred, cat, cat_idx,
                                point_size: int = 4,
                                cmap: str = 'inferno'):
    """
    Density-coloured scatter plot (Gaussian KDE) for a single category.

    • identical axes limits & aspect ratio as the plain scatter
    • no colour-bar
    """
    m = (cat == cat_idx)
    x = y_pred[m]
    y = y_true[m]

    if x.size == 0:        # nothing to show
        return

    # KDE density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # plot least-dense first
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # axis limits (match plain scatter style)
    mn, mx = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = (mx - mn) * 0.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=z, s=point_size, cmap=cmap)

    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    ax.set_aspect('equal', 'box')

    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD",  fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


def plot_scatter_by_cat(y_true, y_pred, cat, title=None):
    """
    Colour‐coded scatter by burn‐category, tight square axes with 5% padding.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    for c, col in cols.items():
        mask = (cat == c)
        if mask.any():
            ax.scatter(y_pred[mask], y_true[mask], c=col, alpha=0.4)
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred - y_true).mean()
        r2   = r2_score(y_true, y_pred)
        ax.set_title(
            f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.2f}",
            fontsize=FONT_LABEL
        )
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD", fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


def plot_bias_hist(y_true, y_pred, title=None, rng=(-100, 300),
                   tick_limit: int = 100):
    """
    Histogram of prediction bias with customised x-tick labelling.

    • x-axis label: “Bias (Days)”
    • tick labels are shown only for values within ±tick_limit
      (spacing stays the same; labels beyond are blanked)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    res = y_pred - y_true
    ax.hist(res, bins=50, range=rng, alpha=0.7)
    ax.axvline(res.mean(), color='k', ls='--', lw=2)
    ax.text(0.02, 0.95, f"N={len(y_true)}",
            transform=ax.transAxes,
            fontsize=FONT_LEGEND, va='top')

    mean, std, r2 = res.mean(), res.std(), r2_score(y_true, y_pred)
    ax.set_title(f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}",
                 fontsize=FONT_LABEL)

    # ── new x-axis label & selective tick labelling ───────────
    ax.set_xlabel("Bias (Days)", fontsize=FONT_LABEL)
    xt = ax.get_xticks()
    ax.set_xticklabels([f'{t:g}' if abs(t) <= tick_limit else '' for t in xt])

    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def plot_top10_features(rf, names):
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(10), imp[idx])
    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [NICE_NAME.get(names[i], names[i]) for i in idx],
        rotation=45, ha='right', fontsize=FONT_TICK
    )
    ax.set_ylabel("Predictor Importance", fontsize=FONT_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(rf, X_val, y_val, names):
    res = permutation_importance(rf, X_val, y_val,
                                 n_repeats=5, random_state=42)
    imp = res.importances_mean
    idx = np.argsort(imp)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(10), imp[idx])
    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [NICE_NAME.get(names[i], names[i]) for i in idx],
        rotation=45, ha='right', fontsize=FONT_TICK
    )
    ax.set_ylabel("Predictor Importance", fontsize=FONT_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

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
        plt.ylabel("Observed DSD")
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
    rf, X, y, cat, names, n_bins: int = 20
):
    imp  = rf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}

    for rank, f_idx in enumerate(top5, start=1):
        fname = names[f_idx]
        pretty = NICE_NAME.get(fname, fname)
        x_all = X[:,f_idx]
        edges = np.linspace(x_all.min(), x_all.max(), n_bins+1)
        centers = 0.5*(edges[:-1]+edges[1:])
        fig, ax = plt.subplots(figsize=(8,4))
        for c,col in cols.items():
            m = cat==c
            if not m.any(): continue
            rho,p = spearmanr(x_all[m], y[m])
            if   p == 0:     p_label = "p=0"
            elif p < 0.01:   p_label = "p<0.01"
            else:            p_label = f"p={p:.2g}"
            ymu, ysd, xv = [], [], []
            for i in range(n_bins):
                sel = m & (x_all>=edges[i]) & (x_all<edges[i+1])
                if not sel.any(): continue
                ymu.append(y[sel].mean())
                ysd.append(y[sel].std(ddof=0))
                xv.append(centers[i])
            if not xv: continue
            ax.errorbar(xv, ymu, yerr=ysd, fmt='o', ms=5, lw=1.5,
                        color=col, ecolor=col, alpha=0.8,
                        label=f"c{c} (ρ={rho:.2f}, {p_label})")
            ax.plot(xv, ymu, '-', color=col, alpha=0.7)
        ax.set_xlabel(pretty, fontsize=FONT_LABEL)
        ax.set_ylabel("Observed DSD (Days)", fontsize=FONT_LABEL)
        ax.set_title(f"Predictor {rank}", fontsize=FONT_LABEL+2)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.legend(fontsize=FONT_LEGEND, loc="best")
        plt.tight_layout()
        plt.show()


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
    axs[0].set_ylabel("DSD (days)")
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
    plt.xlabel("DSD (days)")
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
def dod_map_ca(ds, pix_idx, values, title=None,
               cmap="Blues", vmin=50, vmax=250):
    merc = ccrs.epsg(3857)
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(), lon[pix_idx], lat[pix_idx])[:,:2].T
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6,5))
    add_background(ax, CA_EXTENT, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    sc = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=PIX_SZ, marker="s", transform=merc, zorder=3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, label="DSD (Days)")
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("DSD (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()

def bias_map_ca(ds, pix_idx, y_true, y_pred, title=None):
    merc = ccrs.epsg(3857)
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(), lon[pix_idx], lat[pix_idx])[:,:2].T
    bias = np.clip(y_pred - y_true, -40, 40)
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6,5))
    add_background(ax, CA_EXTENT, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    sc = ax.scatter(x, y, c=bias, cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40),
                    s=PIX_SZ, marker="s", transform=merc, zorder=3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, label="Bias (Days)")
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()



def boxplot_dod_by_elev_veg(y, elev, veg, tag=None):
    edges = [500,1000,1500,2000,2500,3000,3500,4000,4500]
    elev_bin = np.digitize(elev, edges)-1
    vrange, nveg = GLOBAL_VEGRANGE, len(GLOBAL_VEGRANGE)
    data, labels = [], []
    for i in range(len(edges)-1):
        for v in vrange:
            sel = (elev_bin==i)&(veg==v)
            data.append(y[sel])
            labels.append(f"{edges[i]}–{edges[i+1]} m, {VEG_NAMES[v]}")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticklabels(labels, rotation=90, fontsize=FONT_TICK)
    ax.set_xlabel("(Elevation bin, Vegetation Type)", fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def heat_bias_by_elev_veg(y_true, y_pred, elev, veg, tag=None,
                          elev_edges=(500,1000,1500,2000,2500,3000,3500,4000,4500)):
    bias = y_pred - y_true
    elev_bin = np.digitize(elev, elev_edges) - 1

    vrange = GLOBAL_VEGRANGE
    grid = np.full((len(elev_edges)-1, len(vrange)), np.nan)

    for i in range(len(elev_edges)-1):
        for j, v in enumerate(vrange):
            sel = (elev_bin == i) & (veg == v)
            if sel.any():
                grid[i, j] = np.nanmean(bias[sel])

    # ── NEW: round to nearest integer for display & ensure −0 → 0
    grid_display = np.round(grid)
    grid_display[np.isclose(grid_display, 0)] = 0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(np.clip(grid, -15, 15),          # colour limits ±15
                   cmap='seismic_r',
                   vmin=-15, vmax=15,
                   origin='lower', aspect='auto')

    # cell borders + integer labels
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       ec='black', fc='none', lw=0.6))
            
            val = grid_display[i, j]
            if np.isnan(val):
                continue

            # ── new threshold: white if |val| ≥ 8 ───────────────
            txt_color = 'white' if abs(val) >= 8 else 'black'

            ax.text(j, i, f"{int(val):d}",
                    ha='center', va='center',
                    fontsize=FONT_LABEL,
                    color=txt_color)

    ax.set_xticks(range(len(vrange)))
    ax.set_xticklabels([VEG_NAMES[v] for v in vrange],
                       rotation=45, ha='right', fontsize=FONT_TICK)
    ax.set_yticks(range(len(elev_edges)-1))
    ax.set_yticklabels([f"{elev_edges[k]}–{elev_edges[k+1]} m"
                        for k in range(len(elev_edges)-1)],
                       fontsize=FONT_TICK)

    cb = plt.colorbar(im, ax=ax, label="Bias (Days)")
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()



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
                                "Observed DSD – TEST 30 %",
                                fname=None)
    transparent_histogram_by_cat(y_hat_te, cat_te,
                                "Predicted DSD – TEST 30 %",
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
                                "Observed DSD – FULL sample",
                                fname=None)
    transparent_histogram_by_cat(y_hat_all, cat,
                                "Predicted DSD – FULL sample",
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
        plot_scatter(Yv[m], y_hat_all[m], title=None)
        plot_density_scatter_by_cat(Yv, y_hat_all, cat, cat_idx=c)
        r2 = r2_score(Yv[m], y_hat_all[m])
        mean, std = (y_hat_all[m]-Yv[m]).mean(), (y_hat_all[m]-Yv[m]).std()
        plot_bias_hist(
            Yv[m],
            y_hat_all[m],
            title=f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}"
        )

        # observed & predicted DoD maps
        dod_map_ca(
            ds,
            pix_valid[m],
            Yv[m],
            f"Observed DSD – cat {c} (thr={thr})",
            cmap="Blues",
        )
        dod_map_ca(
            ds,
            pix_valid[m],
            y_hat_all[m],
            f"Predicted DSD – cat {c} (thr={thr})",
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

    fig = plt.figure(figsize=(15, 4))
    for j, (key, xlabel) in enumerate([
            ('bias', 'Mean Bias (days)'),
            ('rmse', 'RMSE (days)'),
            ('r2',   'R²')
        ], 1):
        ax = fig.add_subplot(1, 3, j)
        ax.tick_params(axis='both', labelsize=FONT_TICK+2)

        # 1) histograms + dashed means for cats 0,1,3
        for c, col in {0:'black',1:'blue',3:'red'}.items():
            ax.hist(metrics[c][key], bins=10, alpha=.45,
                    color=col, label=f"cat{c}")
            ax.axvline(np.mean(metrics[c][key]), color=col, ls='--', lw=2)
            ax.axvline(orig_stats[c][key],   color=col, ls='-',  lw=2)

        # 2) line-only cat2
        ax.axvline(np.mean(metrics[2][key]), color='grey',
                ls='--', lw=2, label="cat2")
        ax.axvline(orig_stats[2][key], color='grey',
                ls='-', lw=2)

        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        # no title—xlabel is enough
        if j == 1:
            ax.legend(fontsize=FONT_LEGEND+2)
    # no suptitle
    fig.tight_layout()
    plt.show()

    # ── F. feature importance & top‑5 scatter ─────────────────────
    # --- F. feature importance & both Top-5 scatter variants --------
    plot_top10_features(rf, feat_names)
    plot_permutation_importance(rf, X_te, y_te, feat_names)
    plot_top5_feature_scatter(       rf, Xv, Yv, cat, feat_names,
                        f"Top-5 thr={unburned_max_cat}")
    plot_top5_feature_scatter_binned(rf, Xv, Yv, cat, feat_names)
    
    # ── G. Wilcoxon tests on top-5 predictor distributions ─────────────────

    # get the top‐5 feature indices
    top5_idx = np.argsort(rf.feature_importances_)[::-1][:5]
    print("\nWilcoxon rank‐sum tests for top‐5 features (c1,c2,c3 vs c0):")
    for f in top5_idx:
        fname = feat_names[f]
        data0 = Xv[cat == 0, f]
        for c in (1, 2, 3):
            datac = Xv[cat == c, f]
            if datac.size == 0:
                continue
            stat, p = ranksums(data0, datac)
            print(f"  {fname}: cat{c} vs cat0 → statistic={stat:.3f}, p={p:.3e}")

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

    veg_all = ds["VegTyp"].values.ravel(order='C')[ok].astype(int)
    GLOBAL_VEGRANGE = np.unique(veg_all)

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
