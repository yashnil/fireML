#!/usr/bin/env python3
# ============================================================
#  Fire-ML · MLP experiment on final_dataset4.nc
#  70 % unburned-only training → evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
import time, requests
import numpy as np, numpy.random as npr, xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd

# ─── GLOBAL SETTINGS ──────────────────────────────────────────
PIX_SZ      = 1
FONT_LABEL  = 14
FONT_TICK   = 12
FONT_LEGEND = 12
CA_LON_W, CA_LON_E = -124.5, -117.5
CA_LAT_S, CA_LAT_N =   37.0,   42.5

# load state boundaries
STATES = gpd.read_file("data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp") \
             .to_crs(epsg=3857)

# satellite tile check
def _satellite_available(timeout_s=2):
    url = ("https://services.arcgisonline.com/arcgis/rest/services/"
           "World_Imagery/MapServer")
    try:
        requests.head(url, timeout=timeout_s)
        return True
    except:
        return False
USE_SAT = _satellite_available()

# Web-Mercator extent
merc = ccrs.epsg(3857)
x0,y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())
x1,y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())
CA_EXTENT = [x0, y0, x1, y1]

# ─── PLOTTING HELPERS ─────────────────────────────────────────
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
        bias = (y_pred-y_true).mean()
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

def plot_scatter_by_cat(y_true, y_pred, cat, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    for c, col in cols.items():
        m = (cat == c)
        if m.any():
            ax.scatter(y_pred[m], y_true[m], c=col, alpha=0.4)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred-y_true).mean()
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
    ax.set_ylabel("Observed DSD", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.8, label="Counts")
    cbar.ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def plot_bias_hist(y_true, y_pred, title=None, rng=(-100,300)):
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(res, bins=50, range=rng, alpha=0.7)
    ax.axvline(res.mean(), color='k', ls='--', lw=2)
    ax.text(0.02, 0.95, f"N={len(y_true)}", transform=ax.transAxes,
            fontsize=FONT_LEGEND, va='top')
    mean, std, r2 = res.mean(), res.std(), r2_score(y_true, y_pred)
    ax.set_title(
        f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}",
        fontsize=FONT_LABEL
    )
    ax.set_xlabel("Bias (Pred − Obs, days)", fontsize=FONT_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def _add_background(ax, zoom=6):
    ax.set_extent([CA_LON_W,CA_LON_E,CA_LAT_S,CA_LAT_N],
                  crs=ccrs.PlateCarree())
    if USE_SAT:
        try:
            tiler = cimgt.GoogleTiles(style='satellite')
            tiler.request_timeout = 5
            ax.add_image(tiler, zoom)
        except:
            ax.add_feature(cfeature.NaturalEarthFeature(
                "physical","shaded_relief","10m",
                edgecolor="none", facecolor=cfeature.COLORS["land"]), zorder=0)
    else:
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical","shaded_relief","10m",
            edgecolor="none", facecolor=cfeature.COLORS["land"]), zorder=0)
    STATES.boundary.plot(ax=ax, lw=0.6, edgecolor="black", zorder=2)

def dod_map_ca(ds, pix_idx, values, cmap="Blues", vmin=50, vmax=250):
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(),
                                lon[pix_idx], lat[pix_idx])[:,:2].T
    fig, ax = plt.subplots(subplot_kw={"projection":merc}, figsize=(6,5))
    _add_background(ax)
    sc = ax.scatter(x, y, c=values,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=1, marker="s", transform=merc, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, label="DSD (days)")
    cbar.ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def bias_map_ca(ds, pix_idx, y_true, y_pred):
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(),
                                lon[pix_idx], lat[pix_idx])[:,:2].T
    bias = np.clip(y_pred - y_true, -60, 60)
    fig, ax = plt.subplots(subplot_kw={"projection":merc}, figsize=(6,5))
    _add_background(ax)
    sc = ax.scatter(x, y, c=bias,
                    cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60),
                    s=1, marker="s", transform=merc, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, label="Bias (days)")
    cbar.ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

# ─── FEATURE-MATRIX HELPERS ─────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(), 'lat','lon','latitude','longitude',
            'burn_fraction','burn_cumsum'}
    ny = ds.sizes["year"]
    feats = {}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={"year","pixel"}:
            feats[v] = da.values
        elif set(da.dims)=={"pixel"}:
            feats[v] = np.tile(da.values, (ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok

# ─── MLP EXPERIMENT ─────────────────────────────────────────────
def mlp_unburned_experiment(X, y, cat2d, ok, ds, feat_names, unburned_max_cat=0):
    thr = unburned_max_cat
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]
    # 70/30 split per category
    train_idx, test_idx = [], []
    for c in range(thr+1):
        rows = np.where(cat==c)[0]
        if rows.size == 0: continue
        tr, te = train_test_split(rows, test_size=0.30, random_state=42)
        train_idx.append(tr); test_idx.append(te)
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    X_tr, y_tr = Xv[train_idx], Yv[train_idx]
    X_te, y_te = Xv[test_idx],  Yv[test_idx]

    # standardize
    scaler   = StandardScaler().fit(X_tr)
    X_tr_s   = scaler.transform(X_tr)
    X_te_s   = scaler.transform(X_te)
    X_all_s  = scaler.transform(Xv)

    # fit MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64,64),
                       solver='adam', max_iter=1000,
                       random_state=42)
    mlp.fit(X_tr_s, y_tr)

    # A) TRAIN / TEST
    plot_scatter(y_tr, mlp.predict(X_tr_s), f"MLP TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr, mlp.predict(X_tr_s))
    plot_scatter(y_te, mlp.predict(X_te_s), f"MLP TEST  (cat ≤ {thr})")
    plot_bias_hist(y_te, mlp.predict(X_te_s))

    # B) ALL‐DATA
    y_all_hat = mlp.predict(X_all_s)
    plot_scatter_by_cat(Yv, y_all_hat, cat, f"ALL DATA (thr={thr})")
    plot_bias_hist(Yv, y_all_hat)

    # C) PER‐CATEGORY
    pix_full  = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid = pix_full[ok]
    for c in range(4):
        m = (cat == c)
        if not m.any(): continue
        plot_scatter(Yv[m],      y_all_hat[m])
        plot_scatter_density_by_cat(Yv, y_all_hat, cat, cat_idx=c)
        plot_bias_hist(Yv[m],    y_all_hat[m])
        bias_map_ca(ds, pix_valid[m], Yv[m], y_all_hat[m])

if __name__ == "__main__":
    # load
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, int)
    cat2d[bc < 0.25] = 0
    cat2d[(bc >= 0.25)&(bc < 0.50)] = 1
    cat2d[(bc >= 0.50)&(bc < 0.75)] = 2
    cat2d[bc >= 0.75] = 3

    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    mlp_unburned_experiment(X_all, y_all, cat2d, ok, ds, feat_names, unburned_max_cat=0)
    mlp_unburned_experiment(X_all, y_all, cat2d, ok, ds, feat_names, unburned_max_cat=1)
