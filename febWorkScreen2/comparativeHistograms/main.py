import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

# ── sample data ─────────────────────────────────────────────
categories  = ['c0','c1','c2','c3']
bias_data   = {'c0':[ .05,  .06,  .07],
               'c1':[1.08, -.53, -.56],
               'c2':[ .92, -.30, -.36],
               'c3':[4.20, 1.67, 1.53]}

rmse_data   = {'c0':[ 7.55, 11.88, 11.88],
               'c1':[14.16, 13.47, 13.48],
               'c2':[15.12, 14.20, 14.31],
               'c3':[15.88, 13.38, 13.17]}

r2_data     = {'c0':[.946, .866, .866],
               'c1':[.789, .823, .822],
               'c2':[.774, .808, .805],
               'c3':[.718, .799, .805]}

test_labels = ['Experiment 1', 'Experiment 2', 'Experiment 3']
colors      = ['#1f77b4', '#ff7f0e', '#2ca02c']          # E-1, E-2, E-3
metrics     = [bias_data, rmse_data, r2_data]
ylabels     = ["Bias (Days)", "RMSE (Days)", "R²"]

# ── figure canvas:  2 rows × 3 columns ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex='col')
fig.subplots_adjust(hspace=0.35, wspace=0.4)

# ------------------------------------------------------------------
# Row-1  : the original triple bar-chart (Experiments 1-3)
# ------------------------------------------------------------------
row0 = axes[0]
x     = np.arange(len(categories))
width = 0.25

for ax, metric, ylabel in zip(row0, metrics, ylabels):
    for i in range(3):                            # (E-1, E-2, E-3)
        vals = [metric[c][i] for c in categories]
        ax.bar(x + (i - 1) * width, vals, width,
               color=colors[i], alpha=0.9,
               label=test_labels[i] if ax is row0[0] else None)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# ------------------------------------------------------------------
# Row-2  : deltas  (EXP-2  minus  EXP-1)
# ------------------------------------------------------------------
row1 = axes[1]
delta_cmap  = cm.get_cmap('seismic')              # blue → white → red
norm        = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)   # rescales later

for ax, metric, ylabel in zip(row1, metrics, ylabels):
    delta = np.array([metric[c][1] - metric[c][0] for c in categories])
    # normalise colours per-panel so  full span → blue/red
    norm = TwoSlopeNorm(vmin=-abs(delta).max(), vcenter=0,
                        vmax= abs(delta).max())
    bar_colors = delta_cmap(norm(delta))

    ax.bar(x, delta, width*1.5, color=bar_colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel(f"Δ {ylabel}\n(EXP 2 − EXP 1)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# ------------------------------------------------------------------
# single shared legend (row-1 already has the handles)
# ------------------------------------------------------------------
handles, labels = row0[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', bbox_to_anchor=(0.5, 0.04),
           ncol=3, frameon=False, fontsize=14)

plt.show()
