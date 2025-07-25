#!/usr/bin/env python3
# ============================================================
#  Fire-ML evaluation on final_dataset4.nc (Experiment 2)
#  70 %/30 % category-split, whisker plots + uniform-scale maps
#  burn_fraction included in predictors
# ============================================================

# backed up in includeBurnFraction/main.py

import time
import requests
import xarray as xr
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker
from scipy.stats import ranksums, spearmanr
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
from typing import List, Dict

# ────────────────────────────────────────────────────────────
PIX_SZ = 0.5
CA_LON_W, CA_LON_E = -124.5, -117.5   # west, east
CA_LAT_S, CA_LAT_N =   37,   42.5   # south, north
FONT_LABEL  = 14
FONT_TICK   = 12
FONT_LEGEND = 12

# placeholder for veg types that actually occur
GLOBAL_VEGRANGE: np.ndarray = np.array([], dtype=int)
VEG_NAMES = {
    1:"Urban/Built-Up", 2:"Dry Cropland/Pasture", 3:"Irrigated Crop/Pasture",
    4:"Mixed Dry/Irrig.", 5:"Crop/Grass Mosaic", 6:"Crop/Wood Mosaic",
    7:"Grassland", 8:"Shrubland", 9:"Mixed Shrub/Grass",10:"Savanna",
    11:"Deciduous Broadleaf",12:"Deciduous Needleleaf",13:"Evergreen Broadleaf",
    14:"Evergreen Needleleaf",15:"Mixed Forest",16:"Water",
    17:"Herb. Wetland",18:"Wooded Wetland",19:"Barren",20:"Herb. Tundra",
    21:"Wooded Tundra",22:"Mixed Tundra",23:"Bare Ground Tundra",
    24:"Snow/Ice",25:"Playa",26:"Lava",27:"White Sand"
}

STATES_SHP = "data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
STATES = gpd.read_file(STATES_SHP).to_crs(epsg=3857)

# ────────────────────────────────────────────────────────────
#  Human‐readable feature names
# ────────────────────────────────────────────────────────────
NICE_NAME = {}
for season in ("Fall","Winter","Spring","Summer"):
    for feat in ("Temperature","Precipitation","Humidity","Shortwave","Longwave"):
        key = f"aorc{season}{feat}"
        arrow = "↓" if feat=="Shortwave" else ""
        NICE_NAME[key] = f"{season} {feat}{arrow}"
NICE_NAME["peakValue"]      = "Peak SWE"
NICE_NAME["Elevation"]      = "Elevation (m)"
NICE_NAME["slope"]          = "Slope"
NICE_NAME["aspect_ratio"]   = "Aspect Ratio"
NICE_NAME["VegTyp"]         = "Vegetation Type"
NICE_NAME["sweWinter"]      = "Winter SWE"
NICE_NAME["burn_fraction"]  = "Burn Fraction"

# ── keep units for the five top‑predictors ────────────────────────────────
NICE_NAME["peakValue"]               = "Peak SWE (mm)"
NICE_NAME["aorcSpringTemperature"]   = "Spring Temperature (K)"
NICE_NAME["aorcWinterPrecipitation"] = "Winter Precipitation (mm/day)"
NICE_NAME["aorcSpringShortwave"]     = "Spring Shortwave↓ (W/m⁻²)"   # arrow kept

# --- PATCH 1 : helper to strip hard‑coded units ----------------------------
UNITS_TO_STRIP = ["(mm)", "(K)", "(m)", "(mm/day)", "(W/m⁻²)"]

def strip_units(label: str) -> str:
    """Remove unit substrings and tidy spaces/underscores."""
    for u in UNITS_TO_STRIP:
        label = label.replace(u, "")
    return label.replace("_", " ").strip()
# ---------------------------------------------------------------------------

# ────────────────────────────────────────────────────────────
# 0) Timer
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
# 1) Basic plotting helpers

def plot_scatter(y_true, y_pred, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_pred, y_true, alpha=0.3)
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
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
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


def plot_scatter_density_by_cat(y_true, y_pred, cat, cat_idx,
                                point_size: int = 4,
                                cmap: str = "inferno",
                                kde_sample_max: int = 40_000):
    """
    Gaussian-KDE density scatter for a single burn-category.
       • warm colours = higher density
       • NO colour-bar
       • identical limits/aspect as the plain scatter
    """
    sel = (cat == cat_idx)
    x, y = y_pred[sel], y_true[sel]
    if x.size == 0:
        return

    # KDE on a subset if too many points
    if x.size <= kde_sample_max:
        z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    else:
        sub = np.random.choice(x.size, kde_sample_max, replace=False)
        kde = gaussian_kde(np.vstack([x[sub], y[sub]]))
        z   = kde(np.vstack([x, y]))

    order = z.argsort()
    x, y, z = x[order], y[order], z[order]

    mn, mx = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad    = (mx - mn) * 0.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=z, s=point_size, cmap=cmap, rasterized=True)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD",  fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


def plot_bias_hist(y_true, y_pred, title=None,
                   rng=(-100, 300), tick_limit: int = 100):
    """Histogram with ±100-day labelled ticks and x-axis 'Bias (Days)'."""
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=50, range=rng, alpha=0.7)
    ax.axvline(res.mean(), color='k', ls='--', lw=2)
    ax.text(0.02, 0.95, f"N={len(y_true)}",
            transform=ax.transAxes, fontsize=FONT_LEGEND, va='top')

    mean, std, r2 = res.mean(), res.std(), r2_score(y_true, y_pred)
    ax.set_title(f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}",
                 fontsize=FONT_LABEL)

    ax.set_xlabel("Bias (Days)", fontsize=FONT_LABEL)
    xt = ax.get_xticks()
    ax.set_xticklabels([f"{t:g}" if abs(t) <= tick_limit else "" for t in xt])

    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def snowfreq_from_means(wprec_mmday, wtemp_K,
                        mm_per_event=5.0,  winter_days=90.0):
    """
    Heuristic count of snow days per winter using *seasonal* means:
        • scale precip by mm_per_event to turn mm into “events”
        • weight by a logistic temp-based snow fraction
    """
    # logistic weight: ≈1 when T≤0 °C, 0 when T≥4 °C
    weight = snow_fraction_jordan(wtemp_K)
    # proxy in "days" (cap at winter_days for realism)
    proxy = np.minimum(wprec_mmday / mm_per_event * weight, winter_days)
    return proxy.astype(np.float32)

def snow_fraction_jordan(temp_K,
                         t_snow=0.5,    # °C – all-snow below this
                         t_rain=2.5):   # °C – all-rain above this
    """
    Linear rain/snow partitioning after Jordan (1991, SNTHERM.89).

    Parameters
    ----------
    temp_K : ndarray
        Air temperature [K].
    t_snow, t_rain : float
        Lower / upper temperature bounds for fractional snow (°C).

    Returns
    -------
    fsnow : ndarray
        Fraction (0‒1) of precipitation that falls as snow.
        1  below t_snow, 0 above t_rain, linear transition in between.
    """
    t_C = temp_K - 273.15
    # (T_rain − T) / (T_rain − T_snow), then clip to 0‒1 just in case
    fsnow = np.clip((t_rain - t_C) / (t_rain - t_snow), 0.0, 1.0)
    return fsnow.astype(np.float32)

def bias_hist_single(y_true, y_pred,
                     sel, burn_idx, snow_idx,
                     rng=(-100, 100), bins=50, tick_limit=100):
    """
    Draw a single histogram of (y_pred – y_true) where *sel* is True.

    • Title shows only Mean Bias, Bias Std, and R².
    • A descriptive line (burn category & snow band) is printed to stdout.
    """
    if sel.sum() == 0:
        print(f"[SKIP] burn c{burn_idx}, snow {snow_idx}: N=0")
        return

    res   = y_pred[sel] - y_true[sel]
    mean  = res.mean()
    std   = res.std()
    r2    = r2_score(y_true[sel], y_pred[sel])

    # ── console tag for later classification ────────────────
    snow_lbl = {0: "Low", 1: "Moderate", 2: "High"}
    print(f"[PLOT] burn c{burn_idx}, {snow_lbl[snow_idx]} snowfall  "
          f"→ N={sel.sum():5d},  μ={mean:7.2f},  σ={std:6.2f},  R²={r2:5.3f}")

    # ── the plot itself ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=bins, range=rng, alpha=0.7)
    ax.axvline(mean, color='k', ls='--', lw=2)

    ax.set_title(f"Mean Bias={mean:.2f},  Bias Std={std:.2f},  R²={r2:.2f}",
                 fontsize=FONT_LABEL)
    ax.set_xlabel("Bias (Days)", fontsize=FONT_LABEL)
    ax.set_ylabel("Count",       fontsize=FONT_LABEL)

    # trimmed x-tick labels like your original helper
    xt = ax.get_xticks()
    ax.set_xticklabels([f'{t:g}' if abs(t) <= tick_limit else '' for t in xt])

    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


def plot_scatter_by_cat(y_true, y_pred, cat, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    for c, col in cols.items():
        m = cat == c
        if m.any():
            ax.scatter(y_pred[m], y_true[m], c=col, alpha=0.4)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD", fontsize=FONT_LABEL)
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
        [strip_units(NICE_NAME.get(names[i], names[i])) for i in idx],
        rotation=45, ha='right', fontsize=FONT_TICK
    )
    ax.set_ylabel("Predictor Importance", fontsize=FONT_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def plot_permutation_importance(rf, X_val, y_val, names):
    res = permutation_importance(rf, X_val, y_val, n_repeats=5, random_state=42)
    imp = res.importances_mean
    idx = np.argsort(imp)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(10), imp[idx])
    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [strip_units(NICE_NAME.get(names[i], names[i])) for i in idx],
        rotation=45, ha='right', fontsize=FONT_TICK
    )
    ax.set_ylabel("Predictor Importance", fontsize=FONT_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────
# 2) Top-5 binned feature scatter (Spearman + p-value formatting)

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

# ────────────────────────────────────────────────────────────
# 3) Spatial helpers
# ────────────────────────────────────────────────────────────
TILER = cimgt.GoogleTiles(style='satellite')
TILER.request_timeout = 5

_RELIEF = cfeature.NaturalEarthFeature(
    "physical", "shaded_relief", "10m",
    edgecolor="none", facecolor=cfeature.COLORS["land"]
)

def _satellite_available(timeout_s: int = 2) -> bool:
    url = ("https://services.arcgisonline.com/arcgis/rest/services/"
           "World_Imagery/MapServer")
    try:
        requests.head(url, timeout=timeout_s)
        return True
    except (requests.RequestException, socket.error):
        return False

USE_SAT = _satellite_available()
print("[INFO] satellite tiles available:", USE_SAT)

# one-time Web-Mercator extent
merc   = ccrs.epsg(3857)
x0, y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())
x1, y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())
CA_EXTENT = [x0, y0, x1, y1]

# ────────────────────────────────────────────────────────────
#  replace the whole add_background helper with this version
# ────────────────────────────────────────────────────────────
def add_background(ax, extent_merc=None, zoom: int = 6):
    """
    Paint a satellite/shaded-relief backdrop + state borders.

    • The 2nd positional argument (*extent_merc*) is optional and ignored – it
      is kept **only** so calls like  add_background(ax, CA_EXTENT, zoom=6)
      do not raise “multiple values for argument 'zoom'”.
    """
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())

    if USE_SAT:
        try:
            ax.add_image(TILER, zoom, interpolation="nearest")
        except Exception:
            ax.add_feature(_RELIEF, zorder=0)
    else:
        ax.add_feature(_RELIEF, zorder=0)

    STATES.boundary.plot(ax=ax, linewidth=0.6,
                         edgecolor="black", zorder=2)


def dod_map_ca(ds, pix_idx, values, title=None,
               cmap="Blues", vmin=50, vmax=250):
    merc = ccrs.epsg(3857)
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(), lon[pix_idx], lat[pix_idx])[:,:2].T
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6,5))
    add_background(ax, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    sc = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=1, marker="s", transform=merc, zorder=3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, label="DSD (Days)")
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("DSD (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()

# ── helper: aggregate a 1-D sample vector to a per-pixel mean ────────────
def mean_per_pixel(pix_idx: np.ndarray,
                   values:   np.ndarray,
                   n_pix:    int) -> np.ndarray:
    """
    Parameters
    ----------
    pix_idx : 1-D array of pixel IDs, one per sample row
    values  : 1-D array of the same length with the variable to average
    n_pix   : total number of pixels in the dataset (ds.sizes['pixel'])

    Returns
    -------
    mean_val : np.ndarray  (length = n_pix)
        NaN for pixels that have no valid samples.
    """
    mean_val = np.full(n_pix, np.nan, dtype=np.float32)
    count    = np.zeros(n_pix,  dtype=np.uint32)

    for pid, val in zip(pix_idx, values):
        if np.isfinite(val):
            if np.isnan(mean_val[pid]):
                mean_val[pid] = 0.0
            mean_val[pid] += val
            count[pid]    += 1

    mask = count > 0
    mean_val[mask] /= count[mask]
    return mean_val

def bias_map_ca(ds, pix_idx, y_true, y_pred, title=None):
    """Pixel-bias map, colour-limited to ±40 days."""
    merc = ccrs.epsg(3857)
    lat  = ds["latitude"].values.ravel()
    lon  = ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    bias = np.clip(y_pred - y_true, -40, 40)

    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    add_background(ax, CA_EXTENT, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())

    sc = ax.scatter(x, y, c=bias,
                    cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40),
                    s=PIX_SZ, marker="s", transform=merc, zorder=3)

    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────
# 4) Elev×Veg and boxplots
def boxplot_dod_by_cat(y_obs, y_pred, cat, title_prefix):
    cats = [0,1,2,3]
    data_obs = [y_obs[cat==c] for c in cats]
    data_pre = [y_pred[cat==c] for c in cats]
    fig, axs = plt.subplots(1,2,figsize=(10,4), sharey=True)
    axs[0].boxplot(data_obs, showmeans=True); axs[0].set_title(f"{title_prefix} – OBS")
    axs[1].boxplot(data_pre,showmeans=True); axs[1].set_title(f"{title_prefix} – PRED")
    for ax in axs:
        ax.set_xticklabels([f"c{c}" for c in cats]); ax.set_xlabel("Category")
    axs[0].set_ylabel("DSD (days)")
    plt.tight_layout()
    plt.show()

def transparent_histogram_by_cat(vals, cat, title):
    plt.figure(figsize=(6,4))
    rng = (np.nanmin(vals), np.nanmax(vals))
    for c,col in {0:'red',1:'yellow',2:'green',3:'blue'}.items():
        sel = cat==c
        if sel.any():
            plt.hist(vals[sel], bins=40, range=rng, alpha=0.35,
                     label=f"c{c}", density=True, color=col)
    plt.xlabel("DSD (days)"); plt.ylabel("relative freq.")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()


def heat_bias_by_elev_veg(y_true, y_pred, elev, veg,
                          tag=None,                         # tag ignored
                          elev_edges=(500,1000,1500,2000,2500,
                                      3000,3500,4000,4500)):
    bias      = y_pred - y_true
    elev_bin  = np.digitize(elev, elev_edges) - 1
    vrange    = GLOBAL_VEGRANGE

    grid = np.full((len(elev_edges)-1, len(vrange)), np.nan)
    for i in range(len(elev_edges)-1):
        for j, v in enumerate(vrange):
            sel = (elev_bin == i) & (veg == v)
            if sel.any():
                grid[i, j] = np.nanmean(bias[sel])

    # ── display helpers ──────────────────────────────────────────
    grid_lbl        = np.round(grid)
    grid_lbl[np.isclose(grid_lbl, 0)] = 0        # “-0” → “0”
    norm            = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)

    fig, ax = plt.subplots(figsize=(8, 4))
    im  = ax.imshow(np.clip(grid, -15, 15), cmap='seismic_r',
                    norm=norm, origin='lower', aspect='auto')

    # draw cell borders and integer labels
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       ec='black', fc='none', lw=0.6))
            
            val = grid_lbl[i, j]
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

    # ── colour-bar with the full ±15 span ───────────────────────
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_ticks([-15, -10, -5, 0, 5, 10, 15])   # <- forces the labels
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)

    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────
# 5) Feature matrix
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'aorcsummerhumidity','aorcsummerprecipitation',
            'aorcsummerlongwave','aorcsummershortwave','aorcsummertemperature'}
    ny = ds.sizes['year']; feats = {}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}: feats[v]=da.values
        elif set(da.dims)=={'pixel'}:
            feats[v] = np.tile(da.values, (ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds,target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ────────────────────────────────────────────────────────────
# 6) 10 % burn-fraction bins
def eval_bins(y, yp, burn):
    bins = [(i/10, (i+1)/10) for i in range(9)] + [(0.9, None)]
    for lo,hi in bins:
        if hi is None:
            sel = burn>lo; tag=f">{lo*100:.0f}%"
        else:
            sel = (burn>=lo)&(burn<hi); tag=f"{lo*100:.0f}-{hi*100:.0f}%"
        if sel.sum()==0:
            print(f"{tag}: N=0"); continue
        rmse = np.sqrt(mean_squared_error(y[sel], yp[sel]))
        bias= (yp[sel]-y[sel]).mean(); r2 = r2_score(y[sel], yp[sel])
        print(f"{tag}: N={sel.sum():4d}  RMSE={rmse:.2f}  Bias={bias:.2f}  R²={r2:.3f}")

# ────────────────────────────────────────────────────────────
# 7) Main RF experiment (70/30 per cat)
def rf_experiment_nobf(X, y, cat2d, ok, ds, feat_names, snow_cat):
    # flatten
    cat = cat2d.ravel(order='C')[ok]
    Xv, Yv = X[ok], y[ok]

    # 70/30 split per category
    tr_idx, te_idx = [], []
    for c in (0,1,2,3):
        rows = np.where(cat==c)[0]
        if rows.size==0: continue
        tr, te = train_test_split(rows, test_size=0.3, random_state=42)
        tr_idx.append(tr); te_idx.append(te)
    tr_idx = np.concatenate(tr_idx)
    te_idx = np.concatenate(te_idx)
    cat_te = cat[te_idx]

    # train & fit
    X_tr, y_tr = Xv[tr_idx], Yv[tr_idx]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # TEST predictions
    X_te, y_te = Xv[te_idx], Yv[te_idx]
    yhat_te = rf.predict(X_te)

    # A) per-category scatter & bias-hist
    for c in (0,1,2,3):
        m = cat_te==c
        if not m.any(): continue
        plot_scatter(    y_te[m],  yhat_te[m],    title=None)
        plot_scatter_density_by_cat(y_te, yhat_te, cat_te, cat_idx=c)
        plot_bias_hist(  y_te[m],  yhat_te[m],    title=None)

    for burn_c in (0, 1, 2, 3):
        burn_mask = cat_te == burn_c
        if not burn_mask.any():
            continue
        for snow_c in (0, 1, 2):
            mask = burn_mask & (snow_cat[te_idx] == snow_c)
            bias_hist_single(
                y_te, yhat_te,
                sel       = mask,
                burn_idx  = burn_c,
                snow_idx  = snow_c
            )

    # --- sequential Wilcoxon tests on TEST‑set bias -------------
    bias_all_test = yhat_te - y_te
    bias_by_cat   = {c: bias_all_test[cat_te == c]
                     for c in (0, 1, 2, 3) if (cat_te == c).any()}

    print("\n[TEST] Wilcoxon rank‑sum on pixel‑level bias distributions")
    for a, b in [(1, 0), (2, 1), (3, 2)]:
        if a in bias_by_cat and b in bias_by_cat:
            s, p = ranksums(bias_by_cat[a], bias_by_cat[b])
            print(f"  cat {a} vs cat {b} → stat={s:.3f}, p={p:.3g}")
    # -------------------------------------------------------------

    # B) pixel-bias maps (no titles)
    
    # ── NEW: pixel-mean bias maps ───────────────────────────────
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_ok    = pix_full[ok]
    n_pix     = ds.sizes["pixel"]

    # (a) GLOBAL – all test rows pooled
    obs_mean_all  = mean_per_pixel(pix_ok[te_idx], y_te,     n_pix)
    pred_mean_all = mean_per_pixel(pix_ok[te_idx], yhat_te,  n_pix)
    pix_plot      = np.where(~np.isnan(obs_mean_all))[0]
    bias_map_ca(ds, pix_plot,
                obs_mean_all [pix_plot],
                pred_mean_all[pix_plot],
                title=None)              # uniform colour-scale ±40 set in helper

    # (b) PER-CATEGORY maps
    for c in (0, 1, 2, 3):
        mask = cat_te == c
        if not mask.any():
            continue
        obs_mean_c  = mean_per_pixel(pix_ok[te_idx][mask], y_te[mask],     n_pix)
        pred_mean_c = mean_per_pixel(pix_ok[te_idx][mask], yhat_te[mask],  n_pix)
        pix_c       = np.where(~np.isnan(obs_mean_c))[0]
        bias_map_ca(ds, pix_c,
                    obs_mean_c [pix_c],
                    pred_mean_c[pix_c],
                    title=None)

    # C) Elev×Veg per test-category
    for c in (0,1,2,3):
        m = cat_te==c
        if not m.any(): continue
        elev = ds["Elevation"].values.ravel(order='C')[ok][te_idx][m]
        veg  = ds["VegTyp"]   .values.ravel(order='C')[ok][te_idx][m].astype(int)
        heat_bias_by_elev_veg(y_te[m], yhat_te[m], elev, veg, tag=None)

    # D) feature importance
    plot_top10_features(rf, feat_names)
    plot_permutation_importance(rf, Xv, Yv, feat_names)

    # E) top-5 binned
    plot_top5_feature_scatter_binned(rf, Xv, Yv, cat, feat_names)

    # F) down-sampling robustness
    counts = {c:(cat_te==c).sum() for c in (0,1,2,3) if (cat_te==c).any()}
    k = min(counts.values())
    metrics = {c:{'bias':[], 'rmse':[], 'r2':[]} for c in counts}
    for c,n in counts.items():
        idx_c = np.where(cat_te==c)[0]
        for _ in range(100):
            sub = npr.choice(idx_c, size=k, replace=False)
            y_s, p_s = y_te[sub], yhat_te[sub]
            metrics[c]['bias'].append((p_s-y_s).mean())
            metrics[c]['rmse'].append(np.sqrt(mean_squared_error(y_s,p_s)))
            metrics[c]['r2'].append(r2_score(y_s,p_s))

    orig = {c:{
        'bias': np.mean(metrics[c]['bias']),
        'rmse': np.mean(metrics[c]['rmse']),
        'r2':   np.mean(metrics[c]['r2'])
    } for c in metrics}

    fig = plt.figure(figsize=(15,4))
    for j,(key,lab) in enumerate([('bias','Mean Bias (days)'),
                                  ('rmse','RMSE (days)'),
                                  ('r2','R²')], 1):
        ax = fig.add_subplot(1,3,j)
        ax.tick_params(labelsize=FONT_TICK+2)
        for c,col in {0:'black',1:'blue',3:'red'}.items():
            ax.hist(metrics[c][key], bins=10, alpha=0.45, color=col)
            ax.axvline(orig[c][key], color=col, ls='-', lw=2)
        ax.axvline(orig[2][key], color='grey', ls='-', lw=2)
        ax.set_xlabel(lab, fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()

    # G) burn-fraction bins
    bf = ds["burn_fraction"].values.ravel(order='C')[ok][te_idx]
    print("\nPerformance by 10 % burn-fraction bins (Test set):")
    eval_bins(y_te, yhat_te, bf)

    # H) Wilcoxon on top-5 features
    top5 = np.argsort(rf.feature_importances_)[::-1][:5]
    for f in top5:
        pretty = NICE_NAME.get(feat_names[f], feat_names[f])
        v0 = Xv[cat==0, f]
        for c in (1,2,3):
            vc = Xv[cat==c, f]
            if vc.size:
                s,p = ranksums(v0, vc)
                print(f"Feat {pretty} c0 vs c{c}: stat={s:.3f}, p={p:.3g}")

    return rf, bias_all_test, cat_te

# ────────────────────────────────────────────────────────────
# MAIN
if __name__=="__main__":
    log("loading final_dataset5.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")

    # build veg-range
    veg_all = ds["VegTyp"].values.ravel(order='C').astype(int)
    GLOBAL_VEGRANGE = np.unique(veg_all[np.isfinite(veg_all)])

    # compute categories
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, dtype=int)
    cat2d[bc < 0.25] = 0
    cat2d[(bc>=0.25)&(bc<0.50)] = 1
    cat2d[(bc>=0.50)&(bc<0.75)] = 2
    cat2d[bc>=0.75] = 3
    log("categories computed")

    # flatten
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    log("feature matrix ready (burn_fraction included)")

    # ─── snowfall-frequency proxy stratification ─────────────────
    log("building snow-event-frequency proxy from seasonal means …")

    wprec_mmday = ds["aorcWinterPrecipitation"].values * 86_400.0  # mm day⁻¹
    wtemp_K     = ds["aorcWinterTemperature"].values

    snowfreq = snowfreq_from_means(wprec_mmday, wtemp_K)           # (year,pixel)

    snowfreq_vals = snowfreq.ravel(order="C")[ok]                  # 1-D on valid rows
    low_th, high_th = np.percentile(snowfreq_vals, [33, 67])
    snow_cat = np.digitize(snowfreq_vals, [low_th, high_th])       # 0 low | 1 mod | 2 high

    print(f"[snow-proxy] 33 % = {low_th:.1f} days, 67 % = {high_th:.1f} days")

    # run experiment
    rf_model, exp3_bias, exp3_cat = rf_experiment_nobf(
         X_all, y_all,         # features / target
         cat2d, ok, ds, feat_names,
         snow_cat              # NEW positional argument
    )

    import pandas as pd
    from pathlib import Path

    desktop = Path("/Users/yashnilmohanty/Desktop")
    out_dir = desktop / "run3_bias_csv"
    out_dir.mkdir(exist_ok=True)

    for c in (0, 1, 2, 3):
        sel = exp3_cat == c
        if not sel.any():
            continue
        pd.DataFrame({"bias_days": exp3_bias[sel]}) \
          .to_csv(out_dir / f"run3_bias_c{c}.csv",
                  index=False, float_format="%.6g")
        print(f"[WRITE] {out_dir}/run3_bias_c{c}.csv  (N={sel.sum()})")

    print(f"[DONE] All Experiment‑3 bias files saved in: {out_dir}")

    log("ALL DONE (Experiment 3).")
