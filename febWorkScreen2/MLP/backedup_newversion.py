#!/usr/bin/env python3
# ============================================================
#  Fire-ML  ·  MLP experiment on final_dataset4.nc
#  70 % unburned-only training  →  evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
import time, xarray as xr, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ranksums, spearmanr
import numpy.random as npr

import socket, requests
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd

# ── global placeholder for VegTyp values that actually occur ───
GLOBAL_VEGRANGE: np.ndarray = np.array([], dtype=int)
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

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  plotting helpers  (scatter / bias-hist / importance plots)
# ────────────────────────────────────────────────────────────
# 1. plot_scatter helper
def plot_scatter(y_true, y_pred, title=None):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3)
    mn, mx = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
    plt.plot([mn, mx], [mn, mx], 'k--')
    if title is not None:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred - y_true).mean()
        r2   = r2_score(y_true, y_pred)
        plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DSD")
    plt.ylabel("Observed DSD")
    plt.tight_layout()
    plt.show()

# 2. plot_bias_hist helper stays the same:
def plot_bias_hist(y_true, y_pred, title, rng=(-100,300)):
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(res, bins=50, range=rng, alpha=0.7)
    ax.axvline(res.mean(), color='k', ls='--', lw=2)
    ax.text(0.02, 0.95, f"N={len(y_true)}", transform=ax.transAxes,
            fontsize=14, va='top')
    # always draw mean/std/R2 as title
    mean, std, r2 = res.mean(), res.std(), r2_score(y_true, y_pred)
    ax.set_title(f"Mean={mean:.2f}, Std={std:.2f}, R²={r2:.3f}")
    ax.set_xlabel("Bias (Pred − Obs, Days)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title):
    plt.figure(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    for c,col in cols.items():
        m = cat==c
        if m.any():
            plt.scatter(y_pred[m],y_true[m],c=col,alpha=0.4)
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    # no legend, no title
    plt.xlabel("Predicted DSD");  plt.ylabel("Observed DSD")
    plt.tight_layout();  plt.show()

def plot_top10_perm_importance(imp, names, title):
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx])
    plt.xticks(range(10), [names[i] for i in idx], rotation=45, ha='right')
    plt.title(title);  plt.ylabel("Permutation importance (ΔMSE)")
    plt.tight_layout();  plt.show()

# ────────────────────────────────────────────────────────────
#  Spatial helpers (satellite background + per‐pixel plotting)
# ────────────────────────────────────────────────────────────
def _satellite_available(timeout_s: int = 2) -> bool:
    url = ("https://services.arcgisonline.com/arcgis/rest/services/"
           "World_Imagery/MapServer")
    try:
        requests.head(url, timeout=timeout_s); return True
    except: return False

USE_SAT = _satellite_available()

# California Web‐Mercator extent
CA_LON_W, CA_LON_E = -125.0, -117.0
CA_LAT_S, CA_LAT_N =   37.0,   43.0
merc = ccrs.epsg(3857)
x0,y0 = merc.transform_point(CA_LON_W,CA_LAT_S,ccrs.PlateCarree())[:2]
x1,y1 = merc.transform_point(CA_LON_E,CA_LAT_N,ccrs.PlateCarree())[:2]
CA_EXTENT = [x0,y0,x1,y1]

STATES = gpd.read_file("data/cb_2022_us_state_500k/cb_2022_us_state_500k.shp").to_crs(epsg=3857)

def _add_background(ax, zoom=6):
    ax.set_extent([CA_LON_W, CA_LON_E, CA_LAT_S, CA_LAT_N], crs=ccrs.PlateCarree())
    if USE_SAT:
        try:
            tiler = cimgt.GoogleTiles(style='satellite'); tiler.request_timeout=5
            ax.add_image(tiler, zoom, interpolation="nearest")
        except:
            ax.add_feature(cfeature.NaturalEarthFeature("physical","shaded_relief","10m",
                                                        edgecolor="none",facecolor=cfeature.COLORS["land"]), zorder=0)
    else:
        ax.add_feature(cfeature.NaturalEarthFeature("physical","shaded_relief","10m",
                                                    edgecolor="none",facecolor=cfeature.COLORS["land"]), zorder=0)
    STATES.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", zorder=2)

def dod_map_ca(ds, pix_idx, values, title=None, cmap="Blues", vmin=50, vmax=250):
    lat = ds["latitude"].values.ravel(); lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(), lon[pix_idx], lat[pix_idx])[:,:2].T
    fig,ax = plt.subplots(subplot_kw={"projection":merc}, figsize=(6,5))
    _add_background(ax,6)
    ax.set_extent([CA_LON_W,CA_LON_E,CA_LAT_S,CA_LAT_N], crs=ccrs.PlateCarree())
    sc = ax.scatter(x,y,c=values,cmap=cmap,vmin=vmin,vmax=vmax,
                    s=2,marker="s",transform=merc,zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="DSD (days)")
    if title: ax.set_title(title)
    plt.tight_layout(); plt.show()

def bias_map_ca(ds, pix_idx, y_true, y_pred, title=None):
    lat = ds["latitude"].values.ravel(); lon = ds["longitude"].values.ravel()
    x,y = merc.transform_points(ccrs.Geodetic(), lon[pix_idx], lat[pix_idx])[:,:2].T
    bias = np.clip(y_pred - y_true, -60, 60)
    fig,ax = plt.subplots(subplot_kw={"projection":merc}, figsize=(6,5))
    _add_background(ax,6)
    ax.set_extent([CA_LON_W,CA_LON_E,CA_LAT_S,CA_LAT_N], crs=ccrs.PlateCarree())
    sc = ax.scatter(x,y,c=bias,cmap="seismic_r",
                    norm=TwoSlopeNorm(vmin=-60,vcenter=0,vmax=60),
                    s=2,marker="s",transform=merc,zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Bias (Pred − Obs, days)")
    if title: ax.set_title(title)
    plt.tight_layout(); plt.show()

def boxplot_dod_by_elev_veg(y, elev, veg, tag=None):
    edges = [500,1000,1500,2000,2500,3000,3500,4000,4500]
    elev_bin = np.digitize(elev, edges)-1
    uniq_veg = np.unique(veg)
    data, labels = [], []
    for i in range(len(edges)-1):
        for v in uniq_veg:
            sel = (elev_bin==i)&(veg==v)
            data.append(y[sel]); labels.append(f"{edges[i]}–{edges[i+1]} m, {VEG_NAMES[v]}")
    plt.figure(figsize=(12,5))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1,len(labels)+1), labels, rotation=90)
    plt.xlabel("(Elevation bin, VegType)"); plt.ylabel("Raw DSD")
    # no title
    plt.tight_layout(); plt.show()

def heat_bias_by_elev_veg(y_true, y_pred, elev, veg, tag=None,
                          elev_edges=(500,1000,1500,2000,2500,3000,3500,4000,4500)):
    bias = y_pred - y_true
    elev_bin = np.digitize(elev, elev_edges)-1
    veg_range = GLOBAL_VEGRANGE; nveg = len(veg_range)
    grid = np.full((len(elev_edges)-1,nveg), np.nan)
    for i in range(len(elev_edges)-1):
        for j,v in enumerate(veg_range):
            sel = (elev_bin==i)&(veg==v)
            if sel.any(): grid[i,j] = np.nanmean(bias[sel])
    plt.figure(figsize=(8,4))
    im = plt.imshow(grid, cmap='seismic_r', vmin=-60, vmax=60, origin='lower', aspect='auto')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.gca().add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,ec='black',fc='none',lw=0.6))
            if not np.isnan(grid[i,j]):
                plt.text(j,i,f"{grid[i,j]:.0f}",ha='center',va='center',fontsize=6)
    plt.xticks(range(nveg), [VEG_NAMES[v] for v in veg_range], rotation=45, ha='right')
    plt.yticks(range(len(elev_edges)-1), [f"{elev_edges[i]}–{elev_edges[i+1]} m" for i in range(len(elev_edges)-1)])
    plt.colorbar(im, label="Bias (days)")
    # no title
    plt.tight_layout(); plt.show()

def plot_top5_feature_scatter_binned(imp, X, y, cat, names, prefix, n_bins:int=20):
    top5 = np.argsort(imp)[::-1][:5]
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    for f_idx in top5:
        fname = names[f_idx]; x_all = X[:,f_idx]
        edges = np.linspace(x_all.min(),x_all.max(),n_bins+1)
        centers=0.5*(edges[:-1]+edges[1:])
        plt.figure(figsize=(7,5))
        for c,col in cols.items():
            m = cat==c
            if not m.any(): continue
            rho,p = spearmanr(x_all[m], y[m])
            p_label = "p=0" if p==0 else ("p<0.01" if p<0.01 else f"p={p:.2g}")
            ymu, ysd, xv = [],[],[]
            for i in range(n_bins):
                sel = m&(x_all>=edges[i])&(x_all<edges[i+1])
                if not sel.any(): continue
                ymu.append(y[sel].mean()); ysd.append(y[sel].std(ddof=0)); xv.append(centers[i])
            if not xv: continue
            plt.errorbar(xv, ymu, yerr=ysd, fmt='o', ms=4, lw=1, color=col, ecolor=col, alpha=0.8,
                         label=f"cat={c} (ρ={rho:.2f}, {p_label})")
            plt.plot(xv, ymu, '-', color=col, alpha=0.7)
        plt.xlabel(fname); plt.ylabel("Observed DSD")
        plt.title(f"{prefix} (binned)")
        plt.legend()
        plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  feature-matrix helpers  (burn_fraction & burn_cumsum excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude','pixel','year',
            'ncoords_vector','nyears_vector','burn_fraction','burn_cumsum',
            'aorcsummerhumidity','aorcsummerprecipitation',
            'aorcsummerlongwave','aorcsummershortwave','aorcsummertemperature'}
    ny = ds.sizes['year']; feats = {}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}: feats[v]=da.values
        elif set(da.dims)=={'pixel'}: feats[v]=np.tile(da.values,(ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds,target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok= (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

def perm_importance_mlp(model, X_val, y_val, n_repeats=3):
    base = mean_squared_error(y_val,model.predict(X_val))
    imps = np.zeros(X_val.shape[1]); rng=np.random.RandomState(42)
    Xp = X_val.copy()
    for f in range(X_val.shape[1]):
        scores=[]
        for _ in range(n_repeats):
            temp = Xp[:,f].copy()
            Xp[:,f]=rng.permutation(Xp[:,f])
            scores.append(mean_squared_error(y_val, model.predict(Xp)))
            Xp[:,f]=temp
        imps[f]=np.mean(scores)-base
    return imps

def mlp_unburned_experiment(X,y,cat2d,ok,ds,feat_names,unburned_max_cat=0):
    thr = unburned_max_cat
    cat = cat2d.ravel(order='C')[ok]; Xv,yv = X[ok],y[ok]
    log(f"  training on unburned (cat ≤ {thr}): N={(cat<=thr).sum()}")

    # split per category
    train_idx,test_idx=[],[]
    for c in range(thr+1):
        rows = np.where((cat==c)&(cat<=thr))[0]
        if rows.size==0: continue
        tr,te=train_test_split(rows,test_size=0.30,random_state=42)
        train_idx.append(tr); test_idx.append(te)
    train_idx = np.concatenate(train_idx); test_idx = np.concatenate(test_idx)

    X_tr,y_tr = Xv[train_idx], yv[train_idx]
    X_te,y_te = Xv[test_idx],  yv[test_idx]

    # scale
    xsc=StandardScaler(); X_tr_s=xsc.fit_transform(X_tr)
    X_te_s=xsc.transform(X_te); X_all_s=xsc.transform(Xv)

    # fit MLP
    mlp=MLPRegressor(hidden_layer_sizes=(64,64),activation='relu',
                     solver='adam',random_state=42,max_iter=1000)
    mlp.fit(X_tr_s,y_tr)

    # A) TRAIN/TEST plots
    yhat_tr=mlp.predict(X_tr_s)
    plot_scatter(y_tr,yhat_tr,f"MLP TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr,yhat_tr,"TRAIN")

    yhat_te=mlp.predict(X_te_s)
    plot_scatter(y_te,yhat_te,f"MLP TEST (cat ≤ {thr})")
    plot_bias_hist(y_te,yhat_te,"TEST")

    # B) all-data
    yhat_all=mlp.predict(X_all_s)
    plot_scatter_by_cat(yv,yhat_all,cat,f"ALL data (thr={thr})")
    plot_bias_hist(yv,yhat_all,"ALL data")

    # C) per-category scatter & bias-hist
    for c in range(4):
        m = cat == c
        if not m.any():
            continue

        # scatter with NO title and NO legend
        plot_scatter(
            yv[m],
            yhat_all[m],
            title=None
        )

        # bias-hist will draw its own Mean/Std/R²
        plot_bias_hist(
            yv[m],
            yhat_all[m],
            title=None   # ignored by plot_bias_hist's logic
        )


    # C) spatial bias maps
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid= pix_full[ok]
    bias_map_ca(ds,pix_valid,yv,yhat_all,None)
    for c in range(4):
        m = cat==c
        if not m.any(): continue
        bias_map_ca(ds,pix_valid[m],yv[m],yhat_all[m],None)
        elev_c = ds["Elevation"].values.ravel(order='C')[ok][m]
        veg_c  = ds["VegTyp"].values.ravel(order='C')[ok][m]
        boxplot_dod_by_elev_veg(yv[m],elev_c,veg_c)
        heat_bias_by_elev_veg(yv[m],yhat_all[m],elev_c,veg_c)

    # D) importance
    imp=perm_importance_mlp(mlp,X_all_s,yv)
    plot_top10_perm_importance(imp,feat_names,f"Top-10 Perm Imp (thr={thr})")
    plot_top5_feature_scatter_binned(imp, Xv, yv, cat, feat_names, f"Top-5 (thr={thr})")

    # E) Wilcoxon
    bias = yhat_all - yv
    bycat={c:bias[cat==c] for c in range(4) if (cat==c).any()}
    if 0 in bycat:
        print("\nWilcoxon vs cat 0")
        for c in (1,2,3):
            if c in bycat:
                s,p = ranksums(bycat[0],bycat[c])
                print(f"  cat {c} vs 0 → stat={s:.3f}, p={p:.3g}")
    return mlp

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__=="__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc,dtype=int)
    cat2d[bc<0.25]=0; cat2d[(bc>=0.25)&(bc<0.50)]=1
    cat2d[(bc>=0.50)&(bc<0.75)]=2; cat2d[bc>=0.75]=3
    log("categories computed")

    X_all,y_all,feat_names,ok = flatten_nobf(ds,"DOD")
    veg_all = ds["VegTyp"].values.ravel(order='C')[ok].astype(int)
    GLOBAL_VEGRANGE = np.unique(veg_all)

    log("MLP RUN #1: unburned cat 0")
    mlp_unburned_experiment(X_all,y_all,cat2d,ok,ds,feat_names,unburned_max_cat=0)

    log("MLP RUN #2: unburned cat 0+1")
    mlp_unburned_experiment(X_all,y_all,cat2d,ok,ds,feat_names,unburned_max_cat=1)

    log("ALL DONE.")
