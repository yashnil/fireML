#!/usr/bin/env python3
# ============================================================
#  Fire-ML evaluation on final_dataset4.nc (Experiment 2)
#  70 %/30 % category-split, whisker plots + uniform-scale maps
#  burn_fraction **excluded** from predictors
# ============================================================

# backed up in excludeBurnFraction/main.py

import time
import requests
import xarray as xr
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ranksums, spearmanr
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
PIX_SZ = 1
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

def plot_scatter_density_by_cat(y_true, y_pred, cat, cat_idx, gridsize=80):
    """
    Hexbin density scatter, 1:1 line, square axes.
    """
    mask = (cat == cat_idx)
    x, y = y_pred[mask], y_true[mask]
    fig, ax = plt.subplots(figsize=(6,6))
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap='inferno', mincnt=1)
    mn, mx = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = (mx - mn) * 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    ax.set_aspect('equal', 'box')
    ticks = ax.get_yticks()
    ax.set_xticks(ticks)
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD",  fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, label="Counts")
    cb.ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def plot_bias_hist(y_true, y_pred, title=None, rng=(-100,300)):
    fig, ax = plt.subplots(figsize=(6,4))
    res = y_pred - y_true
    ax.hist(res, bins=50, range=rng, alpha=0.7)
    ax.axvline(res.mean(), color='k', ls='--', lw=2)
    ax.text(0.02, 0.95, f"N={len(y_true)}",
            transform=ax.transAxes, fontsize=FONT_LEGEND, va='top')
    mean, std, r2 = res.mean(), res.std(), r2_score(y_true, y_pred)
    ax.set_title(
        f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}",
        fontsize=FONT_LABEL
    )
    ax.set_xlabel("Bias (Pred − Obs, Days)", fontsize=FONT_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_LABEL)
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
        [NICE_NAME.get(names[i], names[i]) for i in idx],
        rotation=45, ha='right', fontsize=FONT_TICK
    )
    ax.set_ylabel("Feature importance", fontsize=FONT_LABEL)
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
        [NICE_NAME.get(names[i], names[i]) for i in idx],
        rotation=45, ha='right', fontsize=FONT_TICK
    )
    ax.set_ylabel("Permutation importance", fontsize=FONT_LABEL)
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
        ax.set_title(f"Feature {rank}", fontsize=FONT_LABEL+2)
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

def add_background(ax, zoom=6):
    """
    Adds a satellite (or shaded-relief fallback) background plus state borders.
    The caller must set map extent beforehand.
    """
    if USE_SAT:
        try:
            ax.add_image(TILER, zoom, interpolation="nearest")
        except Exception as e:
            print("⚠︎ satellite tiles skipped:", e)
            ax.add_feature(_RELIEF, zorder=0)
    else:
        ax.add_feature(_RELIEF, zorder=0)

    # state borders
    states_shp = "data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
    _states = gpd.read_file(states_shp).to_crs(epsg=4326)
    _states.boundary.plot(
        ax=ax,
        linewidth=0.6,
        edgecolor="black",
        zorder=2,
        transform=ccrs.PlateCarree()
    )

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
                    s=1, marker="s", transform=merc, zorder=3)
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
    bias = np.clip(y_pred - y_true, -60, 60)
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6,5))
    add_background(ax, CA_EXTENT, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())
    sc = ax.scatter(x, y, c=bias, cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60),
                    s=1, marker="s", transform=merc, zorder=3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, label="Bias (Days)")
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


def heat_bias_by_elev_veg(y_true, y_pred, elev, veg, tag=None,
                          elev_edges=(500,1000,1500,2000,2500,3000,3500,4000,4500)):
    bias = y_pred - y_true
    elev_bin = np.digitize(elev, elev_edges)-1
    vrange, nveg = GLOBAL_VEGRANGE, len(GLOBAL_VEGRANGE)
    grid = np.full((len(elev_edges)-1, nveg), np.nan)
    for i in range(len(elev_edges)-1):
        for j,v in enumerate(vrange):
            sel = (elev_bin==i)&(veg==v)
            if sel.any(): grid[i,j] = np.nanmean(bias[sel])
    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(grid, cmap='seismic_r', vmin=-60, vmax=60,
                   origin='lower', aspect='auto')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                                       ec='black',fc='none',lw=0.6))
            if not np.isnan(grid[i,j]):
                ax.text(j,i,f"{grid[i,j]:.0f}",
                        ha='center',va='center',fontsize=FONT_LABEL)
    ax.set_xticks(range(nveg))
    ax.set_xticklabels([VEG_NAMES[v] for v in vrange],
                       rotation=45, ha='right', fontsize=FONT_TICK)
    ax.set_yticks(range(len(elev_edges)-1))
    ax.set_yticklabels([f"{elev_edges[i]}–{elev_edges[i+1]} m"
                        for i in range(len(elev_edges)-1)],
                       fontsize=FONT_TICK)
    cb = plt.colorbar(im, ax=ax, label="Bias (Days)")
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────
# 5) Feature matrix
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum',
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
def rf_experiment_nobf(X, y, cat2d, ok, ds, feat_names):
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

    # B) pixel-bias maps (no titles)
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_ok   = pix_full[ok]
    for c in (None, *[0,1,2,3]):
        if c is None:
            idx = pix_ok[te_idx]
            ytrue, ypred = y_te, yhat_te
        else:
            mask = cat_te==c
            idx = pix_ok[te_idx][mask]
            ytrue, ypred = y_te[mask], yhat_te[mask]
        bias_map_ca(ds, idx, ytrue, ypred, title=None)

    # C) Elev×Veg per test-category
    for c in (0,1,2,3):
        m = cat_te==c
        if not m.any(): continue
        elev = ds["Elevation"].values.ravel(order='C')[ok][te_idx][m]
        veg  = ds["VegTyp"]   .values.ravel(order='C')[ok][te_idx][m].astype(int)
        heat_bias_by_elev_veg(y_te[m], yhat_te[m], elev, veg, title=None)

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
    for j,(key,lab) in enumerate([('bias','Mean Bias'),
                                  ('rmse','RMSE'),
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

    return rf

# ────────────────────────────────────────────────────────────
# MAIN
if __name__=="__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

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
    log("feature matrix ready (burn_fraction excluded)")

    # run experiment
    rf_model = rf_experiment_nobf(X_all, y_all, cat2d, ok, ds, feat_names)
    log("ALL DONE (Experiment 2).")
