#!/usr/bin/env python3
# ============================================================
#  Fire‑ML evaluation on final_dataset4.nc
#  (shifted burn‑fraction predictor · full visual diagnostics)
# ============================================================
import time, xarray as xr, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ranksums
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy.random as npr
import cartopy.crs as ccrs, cartopy.feature as cfeature
from typing import List, Dict
import socket, requests, cartopy.io.img_tiles as cimgt, geopandas as gpd

PIX_SZ = 2
CA_LON_W, CA_LON_E = -125.0, -117.0   # west, east
CA_LAT_S, CA_LAT_N =   37.0,   43.0   # south, north

# ────────────────────────────────────────────────────────────
#  0) Timer helper
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  1) Generic plotting helpers
# ────────────────────────────────────────────────────────────
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn, mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label="1:1 line")
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean(); r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD"); plt.ylabel("Observed DoD"); plt.legend()
    plt.tight_layout(); plt.show()

def plot_bias_hist(y_true, y_pred, title, rng=(-100,100)):
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=50, range=rng, alpha=0.7)
    plt.axvline(res.mean(), color='k', ls='--', lw=2)
    plt.title(f"{title}\nMean={res.mean():.2f}, Std={res.std():.2f}")
    plt.xlabel("Bias (Pred‑Obs)"); plt.ylabel("Count")
    plt.tight_layout(); plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title):
    plt.figure(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    for c,col in cols.items():
        m = cat==c
        if m.any():
            plt.scatter(y_pred[m], y_true[m], c=col, alpha=0.4, label=f"cat={c}")
    mn, mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean(); r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD"); plt.ylabel("Observed DoD"); plt.legend()
    plt.tight_layout(); plt.show()

def plot_top10_features(rf, names, title):
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx])
    plt.xticks(range(10), [names[i] for i in idx], rotation=45, ha='right')
    plt.title(title); plt.ylabel("Feature Importance"); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  2) Aggregated Top‑5 whisker scatter
# ────────────────────────────────────────────────────────────
def plot_top5_feature_scatter(rf, X, y, cat, names, prefix):
    imp  = rf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    for f_idx in top5:
        fname = names[f_idx]; feat = X[:,f_idx]
        plt.figure(figsize=(7,5))
        for c,col in cols.items():
            m_c = cat==c
            if not m_c.any(): continue
            r_val = np.corrcoef(feat[m_c], y[m_c])[0,1]
            dod, mean_x, sd_x = [], [], []
            for d in np.unique(y[m_c]):
                m_d = m_c & (y==d)
                dod.append(d)
                mean_x.append(np.mean(feat[m_d]))
                sd_x.append(np.std(feat[m_d]))
            dod   = np.array(dod)
            mean_x= np.array(mean_x)
            sd_x  = np.array(sd_x)
            order = np.argsort(dod)
            dod, mean_x, sd_x = dod[order], mean_x[order], sd_x[order]
            plt.errorbar(mean_x, dod, xerr=sd_x,
                         fmt='o', ms=4, lw=1,
                         color=col, ecolor=col, alpha=0.8,
                         label=f"cat={c} (r={r_val:.2f})")
            plt.plot(mean_x, dod, '-', color=col, alpha=0.7)
        plt.xlabel(fname); plt.ylabel("Observed DoD")
        plt.title(f"{prefix}: {fname}"); plt.legend(); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  3) Spatial helpers
# ────────────────────────────────────────────────────────────


# ── satellite base-map (falls back to NE shaded relief if offline) ──
TILER = cimgt.GoogleTiles(style='satellite')      # or StamenTerrain
TILER.request_timeout = 5

_RELIEF = cfeature.NaturalEarthFeature(
            "physical", "shaded_relief", "10m",
            edgecolor="none", facecolor=cfeature.COLORS["land"])

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

# ── one-time Web-Mercator extent (needed by cartopy) ────────────────
merc   = ccrs.epsg(3857)
x0, y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())
x1, y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())
CA_EXTENT = [x0, y0, x1, y1]

def _add_background(ax, zoom=6):
    """
    Adds a satellite (or shaded-relief fallback) background plus state
    borders.  The caller is expected to have set the map extent already.
    """
    if USE_SAT:
        try:
            ax.add_image(TILER, zoom, interpolation="nearest")
        except Exception as e:
            print("⚠︎ satellite tiles skipped:", e)
            ax.add_feature(_RELIEF, zorder=0)
    else:
        ax.add_feature(_RELIEF, zorder=0)

    # state borders (Natural Earth in EPSG:4326)
    states_shp = "data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
    _states = gpd.read_file(states_shp).to_crs(epsg=4326)
    _states.boundary.plot(ax=ax, linewidth=.6, edgecolor="black",
                          zorder=2, transform=ccrs.PlateCarree())

def _setup_ca_axes(title:str):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125,-113,32,42], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.set_title(title)
    return ax

def dod_map_ca(ds, pix_idx, values, title,
               cmap="Blues", vmin=50, vmax=250):
    """Light-blue → dark-blue DoD map with common 50…250 day scale."""
    merc = ccrs.epsg(3857)
    lat = ds["latitude"].values.ravel(); lon = ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    _add_background(ax, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E,      # ← final hard clip
                    CA_LAT_S, CA_LAT_N],
                   crs=ccrs.PlateCarree())
    sc = ax.scatter(x, y, c=values, cmap=cmap,
                vmin=vmin, vmax=vmax,
                s=PIX_SZ, marker="s", transform=merc, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=.8, label="DoD (days)")
    ax.set_title(title); plt.tight_layout(); plt.show()


def bias_map_ca(ds, pix_idx, y_true, y_pred, title):
    """
    Per-pixel mean bias with a fixed ±60 day diverging colour-bar,
    overlaid on the same background.
    """
    merc = ccrs.epsg(3857)
    bias = np.clip(y_pred - y_true, -60, 60)
    lat = ds["latitude"].values.ravel(); lon = ds["longitude"].values.ravel()
    x, y = merc.transform_points(ccrs.Geodetic(),
                                 lon[pix_idx], lat[pix_idx])[:, :2].T
    fig, ax = plt.subplots(subplot_kw={"projection": merc}, figsize=(6, 5))
    _add_background(ax, zoom=6)
    ax.set_extent([CA_LON_W, CA_LON_E,      # ← final hard clip
                    CA_LAT_S, CA_LAT_N],
                   crs=ccrs.PlateCarree())
    sc = ax.scatter(x, y, c=bias, cmap="seismic",
                norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60),
                s=PIX_SZ, marker="s", transform=merc, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=.8, label="Bias (Pred-Obs, days)")
    ax.set_title(title); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  feature‑matrix helpers  (same as before)
# ────────────────────────────────────────────────────────────
def gather_features(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'aorcsummerhumidity', 'aorcsummerprecipitation',
            'aorcsummerlongwave', 'aorcsummershortwave',
            'aorcsummertemperature'}
    ny = ds.dims['year']; feats={}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}:
            feats[v]=da.values
        elif set(da.dims)=={'pixel'}:
            feats[v]=np.tile(da.values,(ny,1))
    return feats

def flatten(ds,target="DOD"):
    fd=gather_features(ds,target); names=sorted(fd)
    X=np.column_stack([fd[n].ravel(order='C') for n in names])
    y=ds[target].values.ravel(order='C')
    ok=(~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ────────────────────────────────────────────────────────────
#  evaluation by 10 % burn‑fraction bins
# ────────────────────────────────────────────────────────────
def eval_bins(y, yp, burn):
    bins=[(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),
          (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),
          (0.8,0.9),(0.9,None)]
    for lo,hi in bins:
        sel=(burn>lo) if hi is None else ((burn>=lo)&(burn<hi))
        tag=f">{lo*100:.0f}%" if hi is None else f"{lo*100:.0f}-{hi*100:.0f}%"
        if sel.sum()==0:
            print(f"{tag}: N=0")
            continue
        rmse=np.sqrt(mean_squared_error(y[sel],yp[sel]))
        bias=(yp[sel]-y[sel]).mean(); r2=r2_score(y[sel],yp[sel])
        print(f"{tag}: N={sel.sum():5d}  RMSE={rmse:6.2f}  "
              f"Bias={bias:7.2f}  R²={r2:6.3f}")


def heat_bias_by_elev_veg(y_true, y_pred, elev, veg, tag,
                          elev_edges=(500,1000,1500,2000,2500,3000,3500,4000,4500)):
    bias     = y_pred - y_true
    elev_bin = np.digitize(elev, elev_edges) - 1
    veg_range = np.arange(1, 24)         # VegTyp 1 … 23
    grid = np.full((len(elev_edges)-1, len(veg_range)), np.nan)

    for ei in range(len(elev_edges)-1):
        for vv in veg_range:
            sel = (elev_bin == ei) & (veg.astype(int) == vv)
            if sel.any():
                grid[ei, vv-1] = np.nanmean(bias[sel])

    plt.figure(figsize=(8,4))
    im = plt.imshow(grid, cmap='seismic', vmin=-60, vmax=60,
                    origin='lower', aspect='auto')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.gca().add_patch(
                plt.Rectangle((j-0.5, i-0.5), 1, 1,
                              ec='black', fc='none', lw=.6))
            if not np.isnan(grid[i, j]):
                plt.text(j, i, f"{grid[i, j]:.0f}",
                         ha='center', va='center', fontsize=6, color='k')

    plt.xticks(range(len(veg_range)), [f"V{v}" for v in veg_range])
    plt.yticks(range(len(elev_edges)-1),
               [f"{elev_edges[i]}–{elev_edges[i+1]}" for i in range(len(elev_edges)-1)])
    plt.colorbar(im, label="Bias (days)")
    plt.title(tag); plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────
#  6) Main RF experiment
# ────────────────────────────────────────────────────────────
def rf_experiment_nobf(X,y,cat2d,ok,ds,feat_names):

    cat = cat2d.ravel(order='C')[ok]
    Xv, Yv = X[ok], y[ok]

    # 70/30 split per category
    tr_idx, te_idx = [], []
    for c in (0,1,2,3):
        rows=np.where(cat==c)[0]
        if rows.size==0: continue
        tr,te=train_test_split(rows,test_size=0.3,random_state=42)
        tr_idx.append(tr); te_idx.append(te)
    tr_idx=np.concatenate(tr_idx); te_idx=np.concatenate(te_idx)

    X_tr,y_tr = Xv[tr_idx],Yv[tr_idx]
    X_te,y_te = Xv[te_idx],Yv[te_idx]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr,y_tr)

    # --- Train diagnostics
    plot_scatter(y_tr, rf.predict(X_tr), "Train (70 %)")
    plot_bias_hist(y_tr, rf.predict(X_tr), "Bias Hist: Train")

    # --- Test (all categories)
    yhat_te = rf.predict(X_te)
    plot_scatter(y_te, yhat_te, "Test (30 %)")
    plot_bias_hist(y_te, yhat_te, "Bias Hist: Test")

    # --- Per‑category diagnostics + maps
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid= pix_full[ok]
    cat_te   = cat[te_idx]

    for c in (0,1,2,3):
        m = cat_te==c
        if not m.any(): continue
        plot_scatter(y_te[m], yhat_te[m], f"Test cat={c}")
        plot_bias_hist(y_te[m], yhat_te[m], f"Bias Hist cat={c}")
        dod_map_ca(ds, pix_valid[te_idx][m], y_te[m],
                   f"Observed DoD – cat {c}", cmap="Blues")
        dod_map_ca(ds, pix_valid[te_idx][m], yhat_te[m],
                   f"Predicted DoD – cat {c}", cmap="Blues")
        bias_map_ca(ds, pix_valid[te_idx][m], y_te[m], yhat_te[m],
                    f"Pixel Bias – cat {c}")
        
        elev = ds["Elevation"].values.ravel(order='C')[ok][te_idx][m]
        veg  = ds["VegTyp"   ].values.ravel(order='C')[ok][te_idx][m]
        heat_bias_by_elev_veg(y_te[m], yhat_te[m], elev, veg,
                            f"Elev×Veg Bias – cat {c}")

    # --- All‑data colour scatter + bias diagnostics
    plot_scatter_by_cat(y_te, yhat_te, cat_te,
                        "All Test Data – colour by cat")
    plot_bias_hist(y_te, yhat_te, "Bias Hist: ALL Test")
    bias_map_ca(ds, pix_valid[te_idx], y_te, yhat_te,
                "Pixel Bias: ALL Test")

    # --- Wilcoxon (cat1..3 vs cat0)
    bias_vec = yhat_te - y_te
    if (cat_te==0).any():
        for c in (1,2,3):
            if (cat_te==c).any():
                s,p = ranksums(bias_vec[cat_te==0], bias_vec[cat_te==c])
                print(f"Wilcoxon c{c} vs c0: stat={s:.3f}, p={p:.3g}")

    # --- Down‑sampling robustness (merged histogram)
    counts = {c: (cat_te == c).sum() for c in (0, 1, 2, 3) if (cat_te == c).any()}
    k = min(counts.values())                      # uniform sample size
    metrics: Dict[int, Dict[str, List[float]]] = {
        c: {'bias': [], 'rmse': [], 'r2': []} for c in counts
    }

    for c in counts:
        idx_c = np.where(cat_te == c)[0]
        for _ in range(100):
            sub = npr.choice(idx_c, size=k, replace=False)
            yy, pp = y_te[sub], yhat_te[sub]
            metrics[c]['bias'].append(np.mean(pp - yy))
            metrics[c]['rmse'].append(np.sqrt(mean_squared_error(yy, pp)))
            metrics[c]['r2'  ].append(r2_score(yy, pp))

    orig_stats = {c: {
                    'bias': np.mean(yhat_te[cat_te == c] - y_te[cat_te == c]),
                    'rmse': np.sqrt(mean_squared_error(
                                    y_te[cat_te == c], yhat_te[cat_te == c])),
                    'r2'  : r2_score(
                                    y_te[cat_te == c], yhat_te[cat_te == c])}
                for c in counts}

    col_distr = {0: 'black', 1: 'blue', 3: 'red'}   # shown as histograms
    col_line  = {2: 'grey'}                         # dashed-line only

    fig = plt.figure(figsize=(15, 4))
    for j, (key, lab) in enumerate([('bias', 'Mean Bias'),
                                    ('rmse', 'RMSE'),
                                    ('r2',   'R²')], 1):
        ax = fig.add_subplot(1, 3, j)

        # histogram categories (cats 0,1,3)
        for c, col in col_distr.items():
            ax.hist(metrics[c][key], bins=10, alpha=.45, color=col, label=f"cat{c}")
            ax.axvline(np.mean(metrics[c][key]), color=col, ls='--', lw=2)
            ax.axvline(orig_stats[c][key],      color=col, ls='-',  lw=2)

        # dashed-line-only category (cat 2)
        for c, col in col_line.items():
            ax.axvline(np.mean(metrics[c][key]), color=col, ls='--', lw=2, label=f"cat{c}")
            ax.axvline(orig_stats[c][key],       color=col, ls='-',  lw=2)

        ax.set_xlabel(lab); ax.set_title(lab)
        if j == 1: ax.legend()
    fig.suptitle(f"Down-sampling distributions (k={k}) – NoBF")
    fig.tight_layout(); plt.show()

    # --- Feature importance
    plot_top10_features(rf, feat_names, "Top‑10 Feature Importance")
    plot_top5_feature_scatter(rf, X_te, y_te, cat_te, feat_names,
                              prefix="Top‑5")

    # --- 10 % burn‑fraction bins  (always computed)
    bf_full = ds["burn_fraction"].values.ravel(order='C')
    bf_te   = bf_full[ok][te_idx]            # align with y_te / yhat_te
    print("\nPerformance by 10 % burn‑fraction bins (Test set):")
    eval_bins(y_te, yhat_te, bf_te)

    return rf

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__=="__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

    merc = ccrs.epsg(3857)
    x0, y0 = merc.transform_point(CA_LON_W, CA_LAT_S, ccrs.PlateCarree())
    x1, y1 = merc.transform_point(CA_LON_E, CA_LAT_N, ccrs.PlateCarree())
    CA_EXTENT = [x0, y0, x1, y1]
    print("[DEBUG] CA_EXTENT set:", CA_EXTENT)

    bc = ds["burn_cumsum"].values
    cat_2d = np.zeros_like(bc,dtype=int)
    cat_2d[bc<0.25]=0
    cat_2d[(bc>=0.25)&(bc<0.5)]=1
    cat_2d[(bc>=0.5)&(bc<0.75)]=2
    cat_2d[bc>=0.75]=3
    log("categories (c0..c3) computed")

    X_all,y_all,feat_names,ok = flatten(ds,"DOD")
    log("feature matrix ready")

    rf_model_nobf = rf_experiment_nobf(X_all,y_all,cat_2d,ok,ds,feat_names)
    log("ALL DONE.")
