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
from scipy.stats import gaussian_kde

# ─── GLOBAL SETTINGS ──────────────────────────────────────────
PIX_SZ      = 0.5
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

# global VegType range (filled in main)
GLOBAL_VEGRANGE = np.array([], dtype=int)
VEG_NAMES = {
    1:"Urban/Built-Up",2:"Dry Cropland/Pasture",3:"Irrigated Crop/Pasture",
    4:"Mixed Dry/Irrig.",5:"Crop/Grass Mosaic",6:"Crop/Wood Mosaic",
    7:"Grassland",8:"Shrubland",9:"Mixed Shrub/Grass",10:"Savanna",
    11:"Deciduous Broadleaf",12:"Deciduous Needleleaf",13:"Evergreen Broadleaf",
    14:"Evergreen Needleleaf",15:"Mixed Forest",16:"Water",17:"Herb. Wetland",
    18:"Wooded Wetland",19:"Barren",20:"Herb. Tundra",21:"Wooded Tundra",
    22:"Mixed Tundra",23:"Bare Ground Tundra",24:"Snow/Ice",25:"Playa",
    26:"Lava",27:"White Sand"
}

# ─── PLOTTING HELPERS ─────────────────────────────────────────
def plot_scatter(y_true, y_pred, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_pred, y_true, alpha=0.3)
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad); ax.set_ylim(mn - pad, mx + pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred-y_true).mean(); r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.2f}",
                     fontsize=FONT_LABEL)
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout(); plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    for c,col in cols.items():
        m = (cat==c)
        if m.any(): ax.scatter(y_pred[m], y_true[m], c=col, alpha=0.4)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlim(mn - pad, mx + pad); ax.set_ylim(mn - pad, mx + pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred-y_true).mean(); r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.2f}",
                     fontsize=FONT_LABEL)
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout(); plt.show()

def plot_density_scatter_by_cat(y_true, y_pred, cat, cat_idx,
                                point_size: int = 4,
                                cmap: str = "inferno",
                                kde_sample_max: int = 40_000,
                                hist_bins: int = 200):
    """
    Heat-density scatter for one burn category.
    • KDE on ≤ kde_sample_max points, else   KDE on a random subset.
      (much faster than full-set KDE on 100 k+ points)
    • Same limits / aspect as the plain scatter.
    • No colour-bar (spec requirement).
    """
    sel = (cat == cat_idx)
    x   = y_pred[sel]
    y   = y_true[sel]
    N   = x.size
    if N == 0:
        return

    # ---------- density values -----------------------------------------
    if N <= kde_sample_max:
        z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    else:
        sub = np.random.choice(N, size=kde_sample_max, replace=False)
        kde = gaussian_kde(np.vstack([x[sub], y[sub]]))
        z   = kde(np.vstack([x, y]))
        # If that is still too slow, fall back to a 2-D histogram:
        # counts, xe, ye = np.histogram2d(x, y, bins=hist_bins)
        # xi = np.minimum(np.digitize(x, xe) - 1, hist_bins-1)
        # yi = np.minimum(np.digitize(y, ye) - 1, hist_bins-1)
        # z  = counts[xi, yi]

    o   = z.argsort()
    x, y, z = x[o], y[o], z[o]

    # ---------- plotting ------------------------------------------------
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
    """
    Histogram with ±100 day labelled ticks and x-axis “Bias (Days)”.
    """
    residual = y_pred - y_true
    fig, ax  = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=50, range=rng, alpha=0.7)
    ax.axvline(residual.mean(), color='k', ls='--', lw=2)
    ax.text(0.02, 0.95, f"N={len(y_true)}",
            transform=ax.transAxes,
            fontsize=FONT_LEGEND, va='top')

    mean, std, r2 = residual.mean(), residual.std(), r2_score(y_true, y_pred)
    ax.set_title(f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}",
                 fontsize=FONT_LABEL)

    # new x-label + selective tick labels
    ax.set_xlabel("Bias (Days)", fontsize=FONT_LABEL)
    xt = ax.get_xticks()
    ax.set_xticklabels([f"{t:g}" if abs(t) <= tick_limit else "" for t in xt])

    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout()
    plt.show()

def _add_background(ax, zoom=6):
    ax.set_extent([CA_LON_W,CA_LON_E,CA_LAT_S,CA_LAT_N], crs=ccrs.PlateCarree())
    if USE_SAT:
        try:
            tiler = cimgt.GoogleTiles(style='satellite'); tiler.request_timeout=5
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

def dod_map_ca(ds, pix_idx, values,
               cmap="Blues", vmin=50, vmax=250):
    lat = ds["latitude"].values.ravel()
    lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(),
                                lon[pix_idx], lat[pix_idx])[:,:2].T
    fig, ax = plt.subplots(subplot_kw={"projection":merc}, figsize=(6,5))
    _add_background(ax)
    sc = ax.scatter(x, y, c=values,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=1, marker="s", transform=merc, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, label="DSD (Days)")
    cbar.ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout(); plt.show()

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

def bias_map_ca(ds, pix_idx, y_true, y_pred):
    """Bias map capped at ±40 days (colour) and using PIX_SZ symbol size."""
    merc = ccrs.epsg(3857)
    lat  = ds["latitude"].values.ravel()
    lon  = ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    bias = np.clip(y_pred - y_true, -40, 40)

    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    _add_background(ax)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())

    sc = ax.scatter(x, y, c=bias,
                    cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40),
                    s=PIX_SZ, marker="s",
                    transform=merc, zorder=3)

    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()

# ─── ELEV×VEG PLOTS ────────────────────────────────────────────
def boxplot_dod_by_elev_veg(y, elev, veg):
    edges = [500,1000,1500,2000,2500,3000,3500,4000,4500]
    elev_bin = np.digitize(elev, edges) - 1
    data, labels = [], []
    for i in range(len(edges)-1):
        for v in GLOBAL_VEGRANGE:
            sel = (elev_bin==i) & (veg==v)
            data.append(y[sel])
            labels.append(f"{edges[i]}–{edges[i+1]} m, {VEG_NAMES[v]}")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticklabels(labels, rotation=90, fontsize=FONT_TICK)
    ax.set_xlabel("(Elevation bin, VegType)", fontsize=FONT_LABEL)
    ax.set_ylabel("DSD (Days)",              fontsize=FONT_LABEL)
    plt.tight_layout(); plt.show()

def heat_bias_by_elev_veg(y_true, y_pred, elev, veg,
                          elev_edges=(500,1000,1500,2000,2500,
                                      3000,3500,4000,4500)):
    """
    Mean bias per (elev_bin, veg_type) table.
    • colour range ±15
    • values rounded to nearest integer; “-0” coerced to 0
    """
    bias = y_pred - y_true
    elev_bin = np.digitize(elev, elev_edges) - 1
    vrange   = GLOBAL_VEGRANGE

    grid = np.full((len(elev_edges)-1, len(vrange)), np.nan)
    for i in range(len(elev_edges)-1):
        for j, v in enumerate(vrange):
            m = (elev_bin == i) & (veg == v)
            if m.any():
                grid[i, j] = np.nanmean(bias[m])

    display = np.round(grid)
    display[np.isclose(display, 0)] = 0      # −0 → 0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(np.clip(grid, -15, 15), cmap='seismic_r',
                   vmin=-15, vmax=15, origin='lower', aspect='auto')

    # cell borders + integer labels
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       ec='black', fc='none', lw=0.6))
            
            val = display[i, j]
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

    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()

# ─── FEATURE-MATRIX HELPERS ─────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'burn_fraction','burn_cumsum'}
    ny = ds.sizes["year"]; feats = {}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={"year","pixel"}:
            feats[v] = da.values
        elif set(da.dims)=={"pixel"}:
            feats[v] = np.tile(da.values,(ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds,target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ─── MLP EXPERIMENT ─────────────────────────────────────────────
def mlp_unburned_experiment(X, y, cat2d, ok, ds, feat_names, unburned_max_cat=0):
    thr = unburned_max_cat
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]
    # 70/30 split per category
    train_idx, test_idx = [], []
    for c in range(thr+1):
        rows = np.where(cat==c)[0]
        if rows.size==0: continue
        tr, te = train_test_split(rows, test_size=0.30, random_state=42)
        train_idx.append(tr); test_idx.append(te)
    train_idx = np.concatenate(train_idx); test_idx = np.concatenate(test_idx)

    X_tr, y_tr = Xv[train_idx], Yv[train_idx]
    X_te, y_te = Xv[test_idx],  Yv[test_idx]

    # standardize
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    X_all_s = scaler.transform(Xv)

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

    # ------------------------------------------------------------------
    # B) ALL-DATA  (unchanged scatter + hist)
    # ------------------------------------------------------------------
    y_all_hat = mlp.predict(X_all_s)
    plot_scatter_by_cat(Yv, y_all_hat, cat, f"ALL DATA (thr={thr})")
    plot_bias_hist(Yv, y_all_hat)

    # ─── NEW GLOBAL MEAN-BIAS MAP ────────────────────────────────────
    pix_full  = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid = pix_full[ok]                      # pixel id per row
    n_pix     = ds.sizes["pixel"]                 # total number of pixels

    obs_mean_all  = mean_per_pixel(pix_valid, Yv,        n_pix)
    pred_mean_all = mean_per_pixel(pix_valid, y_all_hat, n_pix)

    pix_plot = np.where(~np.isnan(obs_mean_all))[0]      # pixels with data
    bias_map_ca(
        ds,
        pix_plot,
        obs_mean_all [pix_plot],   # y_true  (mean observed DSD)
        pred_mean_all[pix_plot]    # y_pred  (mean predicted DSD)
    )


    # ------------------------------------------------------------------
    # 30 % cat 0 TEST‑SET diagnostic plots  (same styling as 100 % plots)
    # ------------------------------------------------------------------
    #
    # test_idx refers to rows of Xv/Yv;  keep only those that belong to c0
    mask_c0_test = (cat[test_idx] == 0)
    if mask_c0_test.any():
        idx_c0_test = test_idx[mask_c0_test]      # row indices (→ Xv/Yv)
        y_c0_true   = Yv[idx_c0_test]
        y_c0_pred   = y_all_hat[idx_c0_test]

        # 1️⃣  plain scatter – NO title
        plot_scatter(y_c0_true, y_c0_pred, title=None)

        # 1️⃣‑bis  density‑coloured scatter (heat map)
        # helper expects a 'cat' array → give it an all‑zero dummy
        plot_density_scatter_by_cat(
            y_c0_true, y_c0_pred,
            cat=np.zeros_like(y_c0_true, dtype=int),
            cat_idx=0
        )

        # 2️⃣  bias histogram – metrics‑only title
        mean = (y_c0_pred - y_c0_true).mean()
        std  = (y_c0_pred - y_c0_true).std()
        r2   = r2_score(y_c0_true, y_c0_pred)
        plot_bias_hist(
            y_c0_true, y_c0_pred,
            title=f"Mean Bias={mean:.2f}, Bias Std={std:.2f}, R²={r2:.2f}"
        )

        # 3️⃣  pixel‑level mean‑bias map
        obs_c0  = mean_per_pixel(pix_valid[idx_c0_test], y_c0_true, n_pix)
        pred_c0 = mean_per_pixel(pix_valid[idx_c0_test], y_c0_pred, n_pix)
        pix_c0  = np.where(~np.isnan(obs_c0))[0]
        if pix_c0.size:
            bias_map_ca(ds, pix_c0,
                        obs_c0 [pix_c0],
                        pred_c0[pix_c0])

        # 4️⃣  Elev×Veg diagnostics (box‑plot & heat‑map)
        elev_c0 = ds["Elevation"].values.ravel(order="C")[ok][idx_c0_test]
        veg_c0  = ds["VegTyp"   ].values.ravel(order="C")[ok][idx_c0_test]
        boxplot_dod_by_elev_veg(y_c0_true, elev_c0, veg_c0)
        heat_bias_by_elev_veg  (y_c0_true, y_c0_pred, elev_c0, veg_c0)

    # ------------------------------------------------------------------
    # C) PER-CATEGORY + Elev×Veg
    # ------------------------------------------------------------------
    elev_all = ds["Elevation"].values.ravel(order="C")[ok]
    veg_all  = ds["VegTyp"   ].values.ravel(order="C")[ok].astype(int)

    for c in range(4):
        m = (cat == c)
        if not m.any():
            continue

        # ---- existing scatter / density / hist -----------------------
        plot_scatter(Yv[m],      y_all_hat[m])
        plot_density_scatter_by_cat(Yv, y_all_hat, cat, cat_idx=c)
        plot_bias_hist(Yv[m],    y_all_hat[m])

        # ─── NEW PER-CATEGORY MEAN-BIAS MAP ───────────────────────────
        obs_mean_c  = mean_per_pixel(pix_valid[m], Yv[m],        n_pix)
        pred_mean_c = mean_per_pixel(pix_valid[m], y_all_hat[m], n_pix)
        pix_c = np.where(~np.isnan(obs_mean_c))[0]

        bias_map_ca(
            ds,
            pix_c,
            obs_mean_c [pix_c],   # mean observed for this category
            pred_mean_c[pix_c]    # mean predicted for this category
        )

        # ---- existing Elev×Veg diagnostics --------------------------
        boxplot_dod_by_elev_veg(Yv[m], elev_all[m], veg_all[m])
        heat_bias_by_elev_veg  (Yv[m], y_all_hat[m], elev_all[m], veg_all[m])

if __name__ == "__main__":
    # load
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, int)
    cat2d[bc<0.25]            = 0
    cat2d[(bc>=0.25)&(bc<0.50)] = 1
    cat2d[(bc>=0.50)&(bc<0.75)] = 2
    cat2d[bc>=0.75]           = 3

    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    # set the global VegTyp range
    GLOBAL_VEGRANGE = np.unique(ds["VegTyp"].values.ravel(order="C")[ok].astype(int))

    mlp_unburned_experiment(X_all, y_all, cat2d, ok, ds, feat_names, unburned_max_cat=0)
    mlp_unburned_experiment(X_all, y_all, cat2d, ok, ds, feat_names, unburned_max_cat=1)
