#!/usr/bin/env python3
# ============================================================
#  Fire‑ML  ·  shifted burn‑fraction predictor  (final_dataset4)
#  Keeps all legacy visualisations  +  runtime timestamps
# ============================================================
import time, os, netCDF4 as nc, xarray as xr
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ranksums
import numpy.random as npr
from typing import List, Tuple, Optional

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    dt = time.time() - T0
    print(f"[{dt:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  0)  pixel centres
# ────────────────────────────────────────────────────────────
from obtainCoordinates import coords          # (n_pix,2)
N_PIX = len(coords)

# ────────────────────────────────────────────────────────────
#  1)  annual burn‑sum  (identical to legacy loop)
# ────────────────────────────────────────────────────────────
def annual_burn_sum(coords: np.ndarray,
                    nc_pat: str,
                    year: int) -> np.ndarray:
    """
    Sum of 12 monthly bounding‑box means (±0.005°) for MTBS_BurnFraction.
    """
    yearly = np.zeros(N_PIX, dtype=np.float32)
    for mm in range(1, 13):
        fp = nc_pat.format(year=year, month=mm)
        ds  = nc.Dataset(fp, mode="r")
        burn = ds.variables["MTBS_BurnFraction"][:].filled(np.nan).flatten()
        lat  = ds.variables["XLAT_M"][:].flatten()
        lon  = ds.variables["XLONG_M"][:].flatten()
        ok   = np.isfinite(burn) & np.isfinite(lat) & np.isfinite(lon)
        burn, lat, lon = burn[ok], lat[ok], lon[ok]

        for i, (clat, clon) in enumerate(coords):
            in_box = ((lat >= clat-0.005) & (lat <= clat+0.005) &
                      (lon >= clon-0.005) & (lon <= clon+0.005))
            if in_box.any():
                yearly[i] += burn[in_box].mean()
        ds.close()
    return yearly

# ────────────────────────────────────────────────────────────
#  2)  pre‑burn (2001–2002 only)
# ────────────────────────────────────────────────────────────
def compute_pre2003(coords, nc_pat):
    log("computing pre‑burn 2001‑2002 …")
    return (annual_burn_sum(coords, nc_pat, 2001) +
            annual_burn_sum(coords, nc_pat, 2002))

# ────────────────────────────────────────────────────────────
#  3)  cumulative burn & categories
# ────────────────────────────────────────────────────────────
def cumulative_burn(ds: xr.Dataset, pre: np.ndarray):
    bf = ds["burn_fraction"].values
    out = np.zeros_like(bf, dtype=np.float32)
    out[0] = pre + bf[0]
    for y in range(1, bf.shape[0]): out[y] = out[y-1] + bf[y]
    return out

def burn_categories(cum: np.ndarray):
    cat = np.zeros_like(cum, dtype=int)
    cat[cum<0.25]=0; cat[(cum>=0.25)&(cum<0.5)]=1
    cat[(cum>=0.5)&(cum<0.75)]=2; cat[cum>=0.75]=3
    return cat

# ────────────────────────────────────────────────────────────
#  4)  shift burn_fraction back by one year
# ────────────────────────────────────────────────────────────
def shift_burn_fraction(ds0: xr.Dataset,
                        coords, nc_pat) -> xr.Dataset:
    log("shifting burn_fraction back by 1 yr …")
    bf_old = ds0["burn_fraction"].values           # (year,pixel)
    bf_new = np.empty_like(bf_old)
    bf_new[1:] = bf_old[:-1]                       # 2005←2004, … 2018←2017
    bf_new[0]  = annual_burn_sum(coords, nc_pat, 2003)  # 2004←2003
    ds = ds0.copy()
    ds["burn_fraction"] = (("year","pixel"), bf_new)
    return ds

# ────────────────────────────────────────────────────────────
#  5)  feature matrix helpers  (unchanged)
# ────────────────────────────────────────────────────────────
def gather_features(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector'}
    feats, ny = {}, ds.dims['year']
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}: feats[v] = da.values
        elif set(da.dims)=={'pixel'}:      feats[v] = np.tile(da.values,(ny,1))
    return feats

def flatten(ds, target="DOD"):
    fd = gather_features(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok

# ────────────────────────────────────────────────────────────
#  6)  plotting helpers  (original legends preserved)
# ────────────────────────────────────────────────────────────
def plot_scatter(y_true, y_pred, title="Scatter"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--',label='1:1 line')
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean(); r2=r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD"); plt.ylabel("Observed DoD")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_bias_hist(y_true,y_pred,title,x_min=-100,x_max=100):
    res = y_pred-y_true
    plt.figure(figsize=(6,4))
    plt.hist(res,bins=50,range=(x_min,x_max),alpha=0.7)
    plt.axvline(res.mean(),color='k',ls='--',lw=2)
    plt.title(f"{title}\nMean={res.mean():.2f}, Std={res.std():.2f}")
    plt.xlabel("Bias (Pred-Obs)"); plt.ylabel("Count")
    plt.tight_layout(); plt.show()

def plot_scatter_by_cat(y_true,y_pred,cat,title):
    plt.figure(figsize=(6,6))
    colors={0:'red',1:'green',2:'blue',3:'orange'}
    for c,col in colors.items():
        m=cat==c
        if m.any(): plt.scatter(y_pred[m],y_true[m],c=col,alpha=0.4,label=f"cat={c}")
    mn,mx=min(y_pred.min(),y_true.min()),max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--'); rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    bias=(y_pred-y_true).mean(); r2=r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD"); plt.ylabel("Observed DoD"); plt.legend()
    plt.tight_layout(); plt.show()

def plot_top10_features(rf, names, title):
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx]); plt.xticks(range(10),[names[i] for i in idx],
                                             rotation=45,ha='right')
    plt.title(title); plt.ylabel("Feature Importance"); plt.tight_layout(); plt.show()

def plot_top5_feature_scatter(rf,X,y,cat,names,title_prefix):
    imp = rf.feature_importances_; idx=np.argsort(imp)[::-1][:5]
    cols={0:'red',1:'blue',2:'green',3:'purple'}
    for i in idx:
        plt.figure(figsize=(6,5))
        for c,col in cols.items():
            m=cat==c
            plt.scatter(y[m],X[m,i],c=col,alpha=0.4,s=10,label=f"cat={c}")
        r=np.corrcoef(y,X[:,i])[0,1]; plt.legend(title=f"r={r:.2f}")
        plt.xlabel("Observed DOD"); plt.ylabel(names[i])
        plt.title(f"{title_prefix}: {names[i]}"); plt.tight_layout(); plt.show()

def bias_map_simple(ds,pix_idx,y,yp,title):
    n=ds.dims["pixel"]; sum_b=np.zeros(n); cnt=np.zeros(n)
    for p,r in zip(pix_idx,yp-y): sum_b[p]+=r; cnt[p]+=1
    mb=np.full(n,np.nan); mb[cnt>0]=sum_b[cnt>0]/cnt[cnt>0]
    lat=ds["latitude"].values; lon=ds["longitude"].values
    lat1=lat[0] if lat.ndim==2 else lat; lon1=lon[0] if lon.ndim==2 else lon
    vmax=np.nanmax(np.abs(mb)); vmax=1e-6 if not np.isfinite(vmax) or vmax==0 else vmax
    norm=TwoSlopeNorm(-vmax,0,vmax)
    plt.figure(figsize=(7,6))
    plt.scatter(lon1,lat1,c="lightgray",s=5,alpha=0.6)
    sc=plt.scatter(lon1[cnt>0],lat1[cnt>0],c=mb[cnt>0],cmap="bwr",norm=norm,s=10)
    plt.colorbar(sc,shrink=0.8,label="Mean Bias (Pred-Obs)")
    plt.title(title); plt.xlabel("Lon"); plt.ylabel("Lat"); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  7)  evaluation helper – 10 % burn‑fraction bins
# ────────────────────────────────────────────────────────────
def eval_bins(y,yp,burn,bins):
    for lo,hi in bins:
        sel=(burn>lo) if hi is None else ((burn>=lo)&(burn<hi))
        tag=f">{lo*100:.0f}%" if hi is None else f"{lo*100:.0f}-{hi*100:.0f}%"
        if sel.sum()==0: print(f"{tag}: N=0"); continue
        rmse=np.sqrt(mean_squared_error(y[sel],yp[sel]))
        bias=(yp[sel]-y[sel]).mean(); r2=r2_score(y[sel],yp[sel])
        print(f"{tag}: N={sel.sum():4d} RMSE={rmse:6.2f} Bias={bias:7.2f} R²={r2:6.3f}")

# ────────────────────────────────────────────────────────────
#  8)  Random‑Forest experiment (keeps all plots)
# ────────────────────────────────────────────────────────────
def rf_experiment(X_all,y_all,cat2d,ok,ds,feat_names,unburn_cat_max=0):

    # a) split valid data
    cat = cat2d.ravel(order='C')[ok]
    X, y = X_all[ok], y_all[ok]

    # un‑burned mask for training
    m_unburn = cat<=unburn_cat_max
    X_ub, y_ub = X[m_unburn], y[m_unburn]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ub,y_ub,test_size=0.3,random_state=42)

    rf = RandomForestRegressor(n_estimators=100,random_state=42)
    rf.fit(X_tr,y_tr)

    # --- plots: unburned train / test
    plot_scatter(y_tr, rf.predict(X_tr), f"Unburned Train (cat<={unburn_cat_max})")
    plot_bias_hist(y_tr, rf.predict(X_tr),
                   f"Bias Hist: Unburned Train (cat<={unburn_cat_max})")
    y_pred_te = rf.predict(X_te)
    plot_scatter(y_te, y_pred_te, f"Unburned Test (cat<={unburn_cat_max})")
    plot_bias_hist(y_te, y_pred_te,
                   f"Bias Hist: Unburned Test (cat<={unburn_cat_max})")

    # b) evaluate on ALL valid rows
    yhat = rf.predict(X)
    plot_scatter_by_cat(y, yhat, cat, "All Data – colour by cat")
    plot_bias_hist(y, yhat, "Bias Hist: All Valid Data")

    # Wilcoxon
    bias=yhat-y
    for c in (1,2,3):
        if (cat==0).any() and (cat==c).any():
            s,p=ranksums(bias[cat==0],bias[cat==c])
            print(f"Wilcoxon c{c} vs c0: stat={s:.3f}, p={p:.3g}")

    # per‑cat scatter / bias
    for c in (0,1,2,3):
        m=cat==c
        if m.any():
            plot_scatter(y[m],yhat[m],f"Category {c}")
            plot_bias_hist(y[m],yhat[m],f"Bias Hist: cat={c}")

    # colour‑coded pixel bias map
    pix_full=np.tile(np.arange(ds.dims["pixel"]), ds.dims["year"])
    pix_ok  = pix_full[ok]
    bias_map_simple(ds, pix_ok, y, yhat, "Pixel Bias: Simple Grid (All data)")

    # feature importance + top‑5 scatter
    plot_top10_features(rf, feat_names, "Top‑10 Feature Importance")
    plot_top5_feature_scatter(rf, X, y, cat, feat_names, "(All data)")

    # down‑sampling robustness  (100 runs)
    counts={c:(cat==c).sum() for c in (0,1,2,3) if (cat==c).any()}
    k=min(counts.values())
    for c,n in counts.items():
        idx=np.where(cat==c)[0]
        biases,rmses,r2s=[],[],[]
        for _ in range(100):
            sub=npr.choice(idx,size=k,replace=False)
            biases.append((yhat[sub]-y[sub]).mean())
            rmses.append(np.sqrt(mean_squared_error(y[sub],yhat[sub])))
            r2s.append(r2_score(y[sub],yhat[sub]))
        # three histograms
        plt.figure(figsize=(15,4))
        for j,data,col,lab in zip(range(1,4),
                                  [biases,rmses,r2s],
                                  ["steelblue","tomato","seagreen"],
                                  ["Mean Bias","RMSE","R²"]):
            plt.subplot(1,3,j)
            plt.hist(data,bins=10,color=col,alpha=0.85)
            plt.axvline(np.mean(data),color='k',ls='--')
            plt.title(f"{lab}  (cat={c})"); plt.xlabel(lab)
        plt.suptitle(f"Down‑sampling distributions (k={k}, runs=100)")
        plt.tight_layout(); plt.show()

    # 10 % burn‑fraction bins
    if "burn_fraction" in feat_names:
        bf_idx=feat_names.index("burn_fraction")
        bins=[(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),
              (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),
              (0.8,0.9),(0.9,None)]
        print("\nPerformance by 10 % burn‑fraction bins:")
        eval_bins(y, yhat, X[:,bf_idx], bins)

    return rf

# ────────────────────────────────────────────────────────────
#  9)  MAIN
# ────────────────────────────────────────────────────────────
if __name__=="__main__":

    nc_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/" \
             "Merged_BurnArea_{year:04d}{month:02d}.nc"

    # A) pre‑burn
    pre_burn = compute_pre2003(coords, nc_pat)

    # B) original dataset 2004‑2018
    log("loading final_dataset3.nc …")
    ds0 = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset3.nc")

    # C) cumulative burn + categories
    cum = cumulative_burn(ds0, pre_burn)
    ds0["burn_cumsum"] = (("year","pixel"), cum)
    cat_2d = burn_categories(cum)

    # D) shift burn_fraction
    ds = shift_burn_fraction(ds0, coords, nc_pat)
    out_nc = "/Users/yashnilmohanty/Desktop/final_dataset4.nc"
    ds.to_netcdf(out_nc)
    log(f"saved shifted dataset → {out_nc}")

    # E) feature matrix
    X_all, y_all, feat_names, ok = flatten(ds, "DOD")
    log("feature matrix ready")

    # F) RF experiment
    rf_model = rf_experiment(X_all, y_all, cat_2d, ok, ds, feat_names)
    log("ALL DONE.")
