#!/usr/bin/env python3
# ============================================================
#  Fire‑ML evaluation on final_dataset4.nc
#  (burn_fraction *excluded* from predictors)
# ============================================================
import time, os, xarray as xr, numpy as np, matplotlib.pyplot as plt
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
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  plotting helpers  (legends preserved)
# ────────────────────────────────────────────────────────────
def plot_scatter(y_true,y_pred,title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred,y_true,alpha=0.3,label=f"N={len(y_true)}")
    mn,mx=min(y_pred.min(),y_true.min()),max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--',label="1:1 line")
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    bias=(y_pred-y_true).mean(); r2=r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD"); plt.ylabel("Observed DoD"); plt.legend()
    plt.tight_layout(); plt.show()

def plot_bias_hist(y_true,y_pred,title,x_min=-100,x_max=100):
    res=y_pred-y_true
    plt.figure(figsize=(6,4))
    plt.hist(res,bins=50,range=(x_min,x_max),alpha=0.7)
    plt.axvline(res.mean(),color='k',ls='--',lw=2)
    plt.title(f"{title}\nMean={res.mean():.2f}, Std={res.std():.2f}")
    plt.xlabel("Bias (Pred‑Obs)"); plt.ylabel("Count")
    plt.tight_layout(); plt.show()

def plot_scatter_by_cat(y_true,y_pred,cat,title):
    plt.figure(figsize=(6,6))
    colors={0:'red',1:'green',2:'blue',3:'orange'}
    for c,col in colors.items():
        m=cat==c
        if m.any(): plt.scatter(y_pred[m],y_true[m],c=col,alpha=0.4,label=f"cat={c}")
    mn,mx=min(y_pred.min(),y_true.min()),max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    bias=(y_pred-y_true).mean(); r2=r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD"); plt.ylabel("Observed DoD"); plt.legend()
    plt.tight_layout(); plt.show()

def plot_top10_features(rf,names,title):
    imp=rf.feature_importances_; idx=np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10),imp[idx])
    plt.xticks(range(10),[names[i] for i in idx],rotation=45,ha='right')
    plt.title(title); plt.ylabel("Feature Importance"); plt.tight_layout(); plt.show()

def plot_top5_feature_scatter(rf,X,y,cat,names,prefix):
    imp=rf.feature_importances_; idx=np.argsort(imp)[::-1][:5]
    cols={0:'red',1:'blue',2:'green',3:'purple'}
    for i in idx:
        plt.figure(figsize=(6,5))
        for c,col in cols.items():
            m=cat==c
            plt.scatter(y[m],X[m,i],c=col,alpha=0.4,s=10,label=f"cat={c}")
        r=np.corrcoef(y,X[:,i])[0,1]
        plt.legend(title=f"r={r:.2f}")
        plt.xlabel("Observed DOD"); plt.ylabel(names[i])
        plt.title(f"{prefix}: {names[i]}")
        plt.tight_layout(); plt.show()

def bias_map_simple(ds,pix_idx,y,yp,title):
    n=ds.dims["pixel"]; sum_b=np.zeros(n); cnt=np.zeros(n)
    for p,res in zip(pix_idx,yp-y): sum_b[p]+=res; cnt[p]+=1
    mb=np.full(n,np.nan); mb[cnt>0]=sum_b[cnt>0]/cnt[cnt>0]
    lat=ds["latitude"].values; lon=ds["longitude"].values
    lat1=lat[0] if lat.ndim==2 else lat; lon1=lon[0] if lon.ndim==2 else lon
    vmax=np.nanmax(np.abs(mb)); vmax=1e-6 if not np.isfinite(vmax) or vmax==0 else vmax
    norm=TwoSlopeNorm(-vmax,0,vmax)
    plt.figure(figsize=(7,6))
    plt.scatter(lon1,lat1,c="lightgray",s=5,alpha=0.6)
    sc=plt.scatter(lon1[cnt>0],lat1[cnt>0],c=mb[cnt>0],cmap="bwr",norm=norm,s=10)
    plt.colorbar(sc,shrink=0.8,label="Mean Bias (Pred‑Obs)")
    plt.title(title); plt.xlabel("Lon"); plt.ylabel("Lat")
    plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────
#  feature‑matrix helpers  (burn_fraction & burn_cumsum excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds,target="DOD"):
    excl={target.lower(),'lat','lon','latitude','longitude',
          'pixel','year','ncoords_vector','nyears_vector',
          'burn_fraction','burn_cumsum'}
    feats,ny={},ds.dims['year']
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da=ds[v]
        if set(da.dims)=={'year','pixel'}: feats[v]=da.values
        elif set(da.dims)=={'pixel'}:      feats[v]=np.tile(da.values,(ny,1))
    return feats

def flatten_nobf(ds,target="DOD"):
    fd=gather_features_nobf(ds,target); names=sorted(fd)
    X=np.column_stack([fd[n].ravel(order='C') for n in names])
    y=ds[target].values.ravel(order='C')
    ok=(~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ────────────────────────────────────────────────────────────
#  evaluation by 10 % burn‑fraction bins
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
#  main RF experiment  (burn_fraction excluded)
# ────────────────────────────────────────────────────────────
def rf_experiment_nobf(X,y,cat2d,ok,ds,feat_names):

    cat=cat2d.ravel(order='C')[ok]
    Xv,Yv=X[ok],y[ok]

    # 70/30 split per category
    tr_idx,te_idx=[],[]
    for c in (0,1,2,3):
        rows=np.where(cat==c)[0]
        if rows.size==0: continue
        tr,te=train_test_split(rows,test_size=0.3,random_state=42)
        tr_idx.append(tr); te_idx.append(te)
    tr_idx=np.concatenate(tr_idx); te_idx=np.concatenate(te_idx)

    Xtr,Ytr=Xv[tr_idx],Yv[tr_idx]
    Xte,Yte=Xv[te_idx],Yv[te_idx]

    rf=RandomForestRegressor(n_estimators=100,random_state=42)
    rf.fit(Xtr,Ytr)

    # 1) train / test plots
    plot_scatter(Ytr,rf.predict(Xtr),"Train (70 %) – NoBF")
    plot_bias_hist(Ytr,rf.predict(Xtr),"Bias Hist: Train – NoBF")
    Yhat_te=rf.predict(Xte)
    plot_scatter(Yte,Yhat_te,"Test (30 %) – NoBF")
    plot_bias_hist(Yte,Yhat_te,"Bias Hist: Test – NoBF")

    # 2) per‑cat plots
    cat_te=cat[te_idx]
    for c in (0,1,2,3):
        m=cat_te==c
        if m.any():
            plot_scatter(Yte[m],Yhat_te[m],f"Test cat={c} – NoBF")
            plot_bias_hist(Yte[m],Yhat_te[m],f"Bias Hist cat={c} – NoBF")

    # 3) all‑data colour‑coded
    plot_scatter_by_cat(Yte,Yhat_te,cat_te,"All Test Data (colour by cat) – NoBF")

    # 4) Wilcoxon
    bias=Yhat_te-Yte
    for c in (1,2,3):
        if (cat_te==0).any() and (cat_te==c).any():
            s,p=ranksums(bias[cat_te==0],bias[cat_te==c])
            print(f"Wilcoxon c{c} vs c0: stat={s:.3f}, p={p:.3g}")

    # 5) down‑sampling robustness  (100 runs, Bias/RMSE/R²)
    counts={c:(cat_te==c).sum() for c in (0,1,2,3) if (cat_te==c).any()}
    k=min(counts.values()); print(f"\nDown‑sampling: k={k} (100 runs)")
    for c,n in counts.items():
        idx=np.where(cat_te==c)[0]
        biases,rmses,r2s=[],[],[]
        for _ in range(100):
            sub=npr.choice(idx,size=k,replace=False)
            biases.append((Yhat_te[sub]-Yte[sub]).mean())
            rmses.append(np.sqrt(mean_squared_error(Yte[sub],Yhat_te[sub])))
            r2s.append(r2_score(Yte[sub],Yhat_te[sub]))
        print(f"cat={c}: μBias={np.mean(biases):.3f}  μRMSE={np.mean(rmses):.3f}  μR²={np.mean(r2s):.3f}")

        # histograms
        plt.figure(figsize=(15,4))
        for j,data,col,lab in zip((1,2,3),
                                  [biases,rmses,r2s],
                                  ["steelblue","tomato","seagreen"],
                                  ["Mean Bias","RMSE","R²"]):
            plt.subplot(1,3,j); plt.hist(data,bins=10,color=col,alpha=0.85)
            plt.axvline(np.mean(data),color='k',ls='--')
            plt.title(f"{lab} (cat={c})"); plt.xlabel(lab)
        plt.suptitle(f"Down‑sampling distributions (cat={c}, k={k}) – NoBF")
        plt.tight_layout(); plt.show()

    # 6) feature importance + top‑5
    plot_top10_features(rf,feat_names,"Top‑10 Feature Importance – NoBF")
    plot_top5_feature_scatter(rf,Xte,Yte,cat_te,feat_names,"Top‑5 – NoBF")

    '''
    # 7) pixel bias map
    pix_full=np.tile(np.arange(ds.dims["pixel"]), ds.dims["year"])
    bias_map_simple(ds, pix_full[ok][te_idx], Yte, Yhat_te,
                    "Pixel Bias (Test set) – NoBF")
    '''

    # 8) 10 % burn‑fraction bins (eval only)
    bf=ds["burn_fraction"].values.ravel(order='C')[ok][te_idx]
    bins=[(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),
          (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),
          (0.8,0.9),(0.9,None)]
    print("\nPerformance by 10 % burn‑fraction bins:")
    eval_bins(Yte,Yhat_te,bf,bins)

    return rf

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__=="__main__":

    log("loading final_dataset4.nc …")
    ds=xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

    # cumulative‑burn categories (already in file)
    bc=ds["burn_cumsum"].values
    cat_2d=np.zeros_like(bc,dtype=int)
    cat_2d[bc<0.25]=0
    cat_2d[(bc>=0.25)&(bc<0.5)]=1
    cat_2d[(bc>=0.5)&(bc<0.75)]=2
    cat_2d[bc>=0.75]=3

    # feature matrix (burn_fraction & burn_cumsum excluded)
    X_all,y_all,feat_names,ok=flatten_nobf(ds,"DOD")
    log("feature matrix ready (burn_fraction excluded)")
    log(f"Features: {feat_names}")

    # run experiment
    rf_model_nobf=rf_experiment_nobf(X_all,y_all,cat_2d,ok,ds,feat_names)
    log("ALL DONE (NoBF).")
