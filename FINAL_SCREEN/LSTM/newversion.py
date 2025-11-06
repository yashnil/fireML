#!/usr/bin/env python3
# ============================================================
#  Fire-ML · LSTM experiment on final_dataset4.nc
#  70 % unburned-only training → evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
import time, requests
import numpy as np, numpy.random as npr, xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
from scipy.stats import gaussian_kde
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

# ─── GLOBAL SETTINGS ──────────────────────────────────────────
PIX_SZ      = 1.0          # 1.0-km squares (2x larger)
FONT_LABEL  = 14
FONT_TICK   = 12
FONT_LEGEND = 12
CA_LON_W, CA_LON_E = -124.5, -117.5
CA_LAT_S, CA_LAT_N =  37.0,   42.5

# VegType names
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

# load state boundaries
STATES = gpd.read_file("data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp") \
             .to_crs(epsg=3857)

# check for satellite tiles
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
x0,y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())[:2]
x1,y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())[:2]
CA_EXTENT = [x0,y0,x1,y1]

# ─── PLOTTING HELPERS ─────────────────────────────────────────
def plot_scatter(y_true, y_pred, title=None):
    print(f"\n[LSTM] Rendering scatter plot: Predicted vs Observed DSD")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_pred, y_true, alpha=0.3)
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    ax.plot([mn,mx],[mn,mx],'k--', lw=1)
    ax.set_xlim(mn-pad, mx+pad)
    ax.set_ylim(mn-pad, mx+pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred-y_true).mean()
        r2   = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.2f}",
                     fontsize=FONT_LABEL)
    ax.set_xlabel("Predicted DSD", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout(); plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title=None):
    print(f"\n[LSTM] Rendering scatter plot: All categories colored (red=c0, yellow=c1, green=c2, blue=c3)")
    fig, ax = plt.subplots(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    mn, mx = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    pad = (mx - mn) * 0.05
    for c,col in cols.items():
        m = cat==c
        if m.any(): ax.scatter(y_pred[m], y_true[m], c=col, alpha=0.4)
    ax.plot([mn,mx],[mn,mx],'k--', lw=1)
    ax.set_xlim(mn-pad, mx+pad); ax.set_ylim(mn-pad, mx+pad)
    if title:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred-y_true).mean(); r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.2f}",
                     fontsize=FONT_LABEL)
    ax.set_xlabel("Predicted DSD (Days)", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed DSD (Days)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout(); plt.show()

def plot_scatter_density_by_cat(y_true, y_pred, cat, cat_idx,
                                point_size: int = 4,
                                cmap: str = "inferno",
                                kde_sample_max: int = 40_000):
    """
    Heat-density scatter for a single burn-category.
    • Gaussian-KDE on ≤ kde_sample_max points (random subset if more)
    • identical axes / aspect as the plain scatter
    • **no colour-bar**  (per spec)
    """
    print(f"[LSTM] Rendering density scatter plot: Category {cat_idx} with color-coded point density")
    sel = (cat == cat_idx)
    x, y = y_pred[sel], y_true[sel]
    if x.size == 0:
        return

    # ---- density --------------------------------------------------------
    if x.size <= kde_sample_max:
        z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    else:
        sub = np.random.choice(x.size, kde_sample_max, replace=False)
        kde = gaussian_kde(np.vstack([x[sub], y[sub]]))
        z   = kde(np.vstack([x, y]))

    order = z.argsort()
    x, y, z = x[order], y[order], z[order]

    # ---- plot -----------------------------------------------------------
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
    """Histogram with ±100-day labelled ticks and x-axis "Bias (Days)"."""
    print(f"[LSTM] Rendering bias histogram: Distribution of prediction errors")
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

def _add_background(ax, zoom=6, fade_alpha=0.35, fade_color="white", alpha=None):
    """
    Draw satellite (or shaded relief) with a semitransparent white fade sheet
    above it. The data sit above the fade. No title is set here.
    """
    # allow legacy alpha=... calls
    if alpha is not None:
        fade_alpha = alpha

    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N], crs=ccrs.PlateCarree())

    # Base imagery (no alpha on tiles)
    if USE_SAT:
        try:
            tiler = cimgt.GoogleTiles(style='satellite')
            tiler.request_timeout = 5
            ax.add_image(tiler, zoom, interpolation="nearest", zorder=0)
        except Exception:
            ax.add_feature(
                cfeature.NaturalEarthFeature("physical", "shaded_relief", "10m",
                                             edgecolor="none", facecolor=cfeature.COLORS["land"]),
                zorder=0
            )
    else:
        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "shaded_relief", "10m",
                                         edgecolor="none", facecolor=cfeature.COLORS["land"]),
            zorder=0
        )

    # Semitransparent fade sheet above the basemap, below your data
    ax.add_patch(
        Rectangle(
            (CA_LON_W, CA_LAT_S),
            CA_LON_E - CA_LON_W,
            CA_LAT_N - CA_LAT_S,
            transform=ccrs.PlateCarree(),
            facecolor=fade_color, edgecolor="none",
            alpha=fade_alpha, zorder=1
        )
    )

    # State outlines above the fade
    STATES.boundary.plot(ax=ax, lw=0.6, edgecolor="black", zorder=2)


def dod_map_ca(ds, pix_idx, values, cmap="Blues", vmin=50, vmax=250):
    lat = ds["latitude"].values.ravel(); lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(), lon[pix_idx], lat[pix_idx])[:,:2].T
    fig, ax = plt.subplots(subplot_kw={"projection":merc}, figsize=(6,5))
    _add_background(ax)
    sc = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=1, marker="s", transform=merc, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="DSD (Days)")
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
    """Pixel-bias map, colour-limited to ±30 days."""
    print("[LSTM] Rendering spatial bias map: California with pixel-level prediction bias (±30 days)")
    merc = ccrs.epsg(3857)
    lat  = ds["latitude"].values.ravel()
    lon  = ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    bias = np.clip(y_pred - y_true, -30, 30)

    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    _add_background(ax)
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N],
                  crs=ccrs.PlateCarree())

    sc = ax.scatter(x, y, c=bias,
                    cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
                    s=PIX_SZ, marker="s", transform=merc, zorder=3)

    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=FONT_TICK)
    cb.set_label("Bias (Days)", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.show()


# ─── ELEV×VEG PLOTS ────────────────────────────────────────────
def boxplot_dod_by_elev_veg(y, elev, veg):
    print("[LSTM] Rendering boxplot: DSD distribution by elevation bins and vegetation types")
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
    print("[LSTM] Rendering heatmap: Mean bias by elevation bins and vegetation types")
    bias     = y_pred - y_true
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

# ─── FEATURE‐MATRIX HELPERS ─────────────────────────────────────
def gather_features_nobf(ds, target="DSD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'burn_fraction','burn_cumsum'}
    ny = ds.sizes["year"]; feats={}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={"year","pixel"}:
            feats[v]=da.values
        elif set(da.dims)=={"pixel"}:
            feats[v]=np.tile(da.values,(ny,1))
    return feats

def flatten_nobf(ds, target="DSD"):
    fd = gather_features_nobf(ds,target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ─── LSTM EXPERIMENT ────────────────────────────────────────────
def lstm_unburned_experiment(X, y, cat2d, ok, ds, feat_names,
                             unburned_max_cat=0,
                             epochs=100, batch_size=512):
    thr = unburned_max_cat
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]

    # 70/30 train/test per category
    train_idx, test_idx = [], []
    for c in range(thr+1):
        rows = np.where(cat==c)[0]
        if rows.size==0: continue
        tr, te = train_test_split(rows, test_size=0.30, random_state=42)
        train_idx.append(tr); test_idx.append(te)
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    X_tr, y_tr = Xv[train_idx], Yv[train_idx]
    X_te, y_te = Xv[test_idx],  Yv[test_idx]

    # scale
    xsc = StandardScaler().fit(X_tr)
    ysc = StandardScaler().fit(y_tr.reshape(-1,1))
    X_tr_s = xsc.transform(X_tr)
    y_tr_s = ysc.transform(y_tr.reshape(-1,1)).ravel()
    X_te_s = xsc.transform(X_te)
    y_te_s = ysc.transform(y_te.reshape(-1,1)).ravel()
    X_all_s= xsc.transform(Xv)

    # reshape for LSTM
    nfeat = X_tr_s.shape[1]
    def to3D(a): return a.reshape(a.shape[0],1,nfeat)

    # build & train
    model = Sequential([
        LSTM(32, input_shape=(1,nfeat)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(to3D(X_tr_s), y_tr_s,
              validation_data=(to3D(X_te_s), y_te_s),
              epochs=epochs, batch_size=batch_size, verbose=0)

    # helper to invert scale
    def inv(a): return ysc.inverse_transform(a.reshape(-1,1)).ravel()

    # A) TRAIN / TEST
    print(f"\n{'='*60}")
    print(f"[LSTM] Starting plotting sequence for unburned_max_cat={thr}")
    print(f"[LSTM] Training on categories ≤ {thr}, evaluating on all categories")
    print(f"{'='*60}")
    yhat_tr = inv(model.predict(to3D(X_tr_s), batch_size=batch_size).squeeze())
    plot_scatter(y_tr, yhat_tr, f"LSTM TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr, yhat_tr)

    yhat_te = inv(model.predict(to3D(X_te_s), batch_size=batch_size).squeeze())
    plot_scatter(y_te, yhat_te, f"LSTM TEST  (cat ≤ {thr})")
    plot_bias_hist(y_te, yhat_te)

    # B) ALL-DATA
    yhat_all = inv(model.predict(to3D(X_all_s), batch_size=batch_size).squeeze())

    # ─── NEW: pixel-wise means over all available years ──────────────
    pix_full  = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid = pix_full[ok]                      # pixel ID per sample
    n_pix     = ds.sizes["pixel"]

    obs_mean_all  = mean_per_pixel(pix_valid, Yv,        n_pix)
    pred_mean_all = mean_per_pixel(pix_valid, yhat_all,  n_pix)

    pix_plot = np.where(~np.isnan(obs_mean_all))[0]     # pixels with data
    bias_map_ca(ds,
                pix_plot,
                obs_mean_all [pix_plot],    # mean observed DSD
                pred_mean_all[pix_plot])    # mean predicted DSD
    

    # ------------------------------------------------------------------
    # 30 % cat 0 TEST‑SET diagnostic plots  (style must match 100 % plots)
    # ------------------------------------------------------------------
    mask_c0_test = (cat[test_idx] == 0)          # rows that are test‑set AND c0
    if mask_c0_test.any():
        idx_c0_test = test_idx[mask_c0_test]     # → row indices for Xv/Yv
        y_c0_true   = Yv[idx_c0_test]
        y_c0_pred   = yhat_all[idx_c0_test]

        # 1️⃣  plain scatter — NO title
        plot_scatter(y_c0_true, y_c0_pred, title=None)

        # 1️⃣‑bis  density‑coloured scatter (heat map, no colour‑bar)
        plot_scatter_density_by_cat(
            y_c0_true, y_c0_pred,
            cat=np.zeros_like(y_c0_true, dtype=int),   # dummy all‑zero category
            cat_idx=0
        )

        # 2️⃣  bias histogram (metrics‑only title)
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
    
    plot_scatter_by_cat(Yv, yhat_all, cat, f"ALL DATA (thr={thr})")
    plot_bias_hist(Yv, yhat_all)

    # C) PER-CATEGORY + Elev×Veg
    pix_full  = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid = pix_full[ok]
    elev_all  = ds["Elevation"].values.ravel(order="C")[ok]
    veg_all   = ds["VegTyp"].values.ravel(order="C")[ok].astype(int)
    for c in range(4):
        m = (cat==c)
        if not m.any(): continue
        plot_scatter(Yv[m],      yhat_all[m])
        plot_scatter_density_by_cat(Yv, yhat_all, cat, cat_idx=c)
        plot_bias_hist(Yv[m],    yhat_all[m])

        # ─── NEW: pixel-wise means for **this** burn category ────────
        obs_mean_c  = mean_per_pixel(pix_valid[m], Yv[m],        n_pix)
        pred_mean_c = mean_per_pixel(pix_valid[m], yhat_all[m],  n_pix)
        pix_c = np.where(~np.isnan(obs_mean_c))[0]

        bias_map_ca(ds,
                    pix_c,
                    obs_mean_c [pix_c],     # mean observed in cat c
                    pred_mean_c[pix_c])     # mean predicted in cat c
        
        boxplot_dod_by_elev_veg(Yv[m], elev_all[m], veg_all[m])
        heat_bias_by_elev_veg   (Yv[m], yhat_all[m], elev_all[m], veg_all[m])

if __name__=="__main__":
    log = lambda msg: print(f"[{time.time():.1f}] {msg}", flush=True)

    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
    # categories
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, int)
    cat2d[bc<0.25]            = 0
    cat2d[(bc>=0.25)&(bc<0.50)] = 1
    cat2d[(bc>=0.50)&(bc<0.75)] = 2
    cat2d[bc>=0.75]           = 3
    log("categories computed")

    # build feature matrix
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DSD")
    # set global veg range
    GLOBAL_VEGRANGE = np.unique(ds["VegTyp"].values.ravel(order="C")[ok].astype(int))

    # RUN for cat 0 and cat 0+1
    log("LSTM RUN #1: unburned cat 0")
    lstm_unburned_experiment(X_all, y_all, cat2d, ok, ds, feat_names,
                             unburned_max_cat=0, epochs=100, batch_size=512)
    log("LSTM RUN #2: unburned cat 0+1")
    lstm_unburned_experiment(X_all, y_all, cat2d, ok, ds, feat_names,
                             unburned_max_cat=1, epochs=100, batch_size=512)
    log("ALL DONE.")
