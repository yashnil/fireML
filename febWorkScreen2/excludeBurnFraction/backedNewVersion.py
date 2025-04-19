#!/usr/bin/env python3
# ============================================================
#  Fire‑ML evaluation on final_dataset4.nc
#  (burn_fraction *excluded* from predictors)
#  70 %/30 % category‑split, whisker plots + uniform‑scale maps
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
def _setup_ca_axes(title:str):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125,-113,32,42], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.set_title(title)
    return ax

def dod_map_ca(ds, pix_idx, values, title, cmap="magma", vmin=50, vmax=250):
    lat = ds["latitude"].values; lon = ds["longitude"].values
    lat1 = lat[0] if lat.ndim==2 else lat
    lon1 = lon[0] if lon.ndim==2 else lon
    ax = _setup_ca_axes(title)
    sc = ax.scatter(lon1[pix_idx], lat1[pix_idx],
                    c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=10, alpha=0.9,
                    transform=ccrs.PlateCarree())
    plt.colorbar(sc, ax=ax, shrink=0.8, label="DoD (days)")
    plt.tight_layout(); plt.show()

def bias_map_ca(ds, pix_idx, y_true, y_pred, title):
    n_pix = ds.sizes["pixel"]
    sum_b = np.zeros(n_pix); cnt = np.zeros(n_pix)
    for p,res in zip(pix_idx, y_pred-y_true):
        sum_b[p]+=res; cnt[p]+=1
    mean_b = np.full(n_pix,np.nan)
    valid  = cnt>0
    mean_b[valid] = sum_b[valid]/cnt[valid]
    lat = ds["latitude"].values; lon = ds["longitude"].values
    lat1=lat[0] if lat.ndim==2 else lat; lon1=lon[0] if lon.ndim==2 else lon
    vmax = np.nanmax(np.abs(mean_b[valid])) if valid.any() else 1e-6
    vmax = vmax if np.isfinite(vmax) and vmax>0 else 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    ax = _setup_ca_axes(title)
    ax.scatter(lon1,lat1,c="lightgray",s=4,alpha=0.4,transform=ccrs.PlateCarree())
    sc = ax.scatter(lon1[valid], lat1[valid], c=mean_b[valid],
                    cmap="bwr", norm=norm,
                    s=10, alpha=0.9, transform=ccrs.PlateCarree())
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Mean Bias (Pred‑Obs)")
    plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  4) Feature matrix (burn_fraction & burn_cumsum excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum',
            'aorcsummerhumidity','aorcsummerprecipitation',
            'aorcsummerlongwave','aorcsummershortwave','aorcsummertemperature'}
    ny = ds.dims['year']; feats={}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}:
            feats[v]=da.values
        elif set(da.dims)=={'pixel'}:
            feats[v]=np.tile(da.values,(ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd=gather_features_nobf(ds,target); names=sorted(fd)
    X=np.column_stack([fd[n].ravel(order='C') for n in names])
    y=ds[target].values.ravel(order='C')
    ok=(~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ────────────────────────────────────────────────────────────
#  5) 10 % burn‑fraction bins
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
    plot_scatter(y_tr, rf.predict(X_tr), "Train (70 %) – NoBF")
    plot_bias_hist(y_tr, rf.predict(X_tr), "Bias Hist: Train – NoBF")

    # --- Test (all categories)
    yhat_te = rf.predict(X_te)
    plot_scatter(y_te, yhat_te, "Test (30 %) – NoBF")
    plot_bias_hist(y_te, yhat_te, "Bias Hist: Test – NoBF")

    # --- Per‑category diagnostics + maps
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    pix_valid= pix_full[ok]
    cat_te   = cat[te_idx]

    for c in (0,1,2,3):
        m = cat_te==c
        if not m.any(): continue
        plot_scatter(y_te[m], yhat_te[m], f"Test cat={c} – NoBF")
        plot_bias_hist(y_te[m], yhat_te[m], f"Bias Hist cat={c} – NoBF")
        dod_map_ca(ds, pix_valid[te_idx][m], y_te[m],
                   f"Observed DoD – cat {c}", cmap="magma")
        dod_map_ca(ds, pix_valid[te_idx][m], yhat_te[m],
                   f"Predicted DoD – cat {c}", cmap="magma")
        bias_map_ca(ds, pix_valid[te_idx][m], y_te[m], yhat_te[m],
                    f"Pixel Bias – cat {c}")

    # --- All‑data colour scatter + bias diagnostics
    plot_scatter_by_cat(y_te, yhat_te, cat_te,
                        "All Test Data – colour by cat (NoBF)")
    plot_bias_hist(y_te, yhat_te, "Bias Hist: ALL Test – NoBF")
    bias_map_ca(ds, pix_valid[te_idx], y_te, yhat_te,
                "Pixel Bias: ALL Test – NoBF")

    # --- Wilcoxon (cat1..3 vs cat0)
    bias_vec = yhat_te - y_te
    if (cat_te==0).any():
        for c in (1,2,3):
            if (cat_te==c).any():
                s,p = ranksums(bias_vec[cat_te==0], bias_vec[cat_te==c])
                print(f"Wilcoxon c{c} vs c0: stat={s:.3f}, p={p:.3g}")

    # --- Down‑sampling robustness (merged histogram)
    counts = {c:(cat_te==c).sum() for c in (0,1,2,3) if (cat_te==c).any()}
    k = min(counts.values()); ref_cat=min(counts, key=counts.get)
    metrics: Dict[int,Dict[str,List[float]]] = {c:{'bias':[],'rmse':[],'r2':[]} for c in counts}
    for c in counts:
        idx_c=np.where(cat_te==c)[0]
        for _ in range(100):
            sub=npr.choice(idx_c,size=k,replace=False)
            yy,pp = y_te[sub], yhat_te[sub]
            metrics[c]['bias'].append(pp.mean()-yy.mean())
            metrics[c]['rmse'].append(np.sqrt(mean_squared_error(yy,pp)))
            metrics[c]['r2'  ].append(r2_score(yy,pp))
    colours={0:'red',1:'yellow',3:'blue'}
    fig=plt.figure(figsize=(15,4))
    for j,(key,lab) in enumerate(zip(['bias','rmse','r2'],["Mean Bias","RMSE","R²"]),1):
        ax=fig.add_subplot(1,3,j)
        for c,col in colours.items():
            if c in metrics:
                ax.hist(metrics[c][key], bins=10, alpha=0.45, color=col, label=f"cat{c}")
        ref_val = np.mean(metrics[ref_cat][key])
        ax.axvline(ref_val,color='grey',ls='--',lw=2,label=f"cat{ref_cat} mean")
        ax.set_xlabel(lab); ax.set_ylabel("count"); ax.set_title(lab)
        if j==1: ax.legend()
    fig.suptitle(f"Down‑sampling distributions (k={k}) – NoBF")
    fig.tight_layout(); plt.show()

    # --- Feature importance
    plot_top10_features(rf, feat_names, "Top‑10 Feature Importance – NoBF")
    plot_top5_feature_scatter(rf, X_te, y_te, cat_te, feat_names,
                              prefix="Top‑5 (NoBF)")

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

    bc = ds["burn_cumsum"].values
    cat_2d = np.zeros_like(bc,dtype=int)
    cat_2d[bc<0.25]=0
    cat_2d[(bc>=0.25)&(bc<0.5)]=1
    cat_2d[(bc>=0.5)&(bc<0.75)]=2
    cat_2d[bc>=0.75]=3
    log("categories (c0..c3) computed")

    X_all,y_all,feat_names,ok = flatten_nobf(ds,"DOD")
    log("feature matrix ready (burn_fraction excluded)")

    rf_model_nobf = rf_experiment_nobf(X_all,y_all,cat_2d,ok,ds,feat_names)
    log("ALL DONE (NoBF).")
