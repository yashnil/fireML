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

x     = np.arange(len(categories))
width = 0.25                  # bar width

# ===========================================================
#  FIGURE 1  : Experiments 1-3
# ===========================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 3.8), sharex=True)
fig1.subplots_adjust(wspace=0.4)

for ax, metric, ylabel in zip(axes1, metrics, ylabels):
    for i in range(3):                            # (E-1, E-2, E-3)
        vals = [metric[c][i] for c in categories]
        ax.bar(x + (i - 1) * width, vals, width,
               color=colors[i], alpha=0.9,
               label=test_labels[i] if ax is axes1[0] else None)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# shared legend underneath figure 1
handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels,
            loc='lower center', bbox_to_anchor=(0.5, -0.12),
            ncol=3, frameon=False, fontsize=14)

fig1.tight_layout()
plt.show()

# ===========================================================
#  FIGURE 2  : Δ  (EXP 2 − EXP 1)
# ===========================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 3.8), sharex=True)
fig2.subplots_adjust(wspace=0.4)

delta_cmap = cm.get_cmap('seismic')              # blue → white → red

for ax, metric, ylabel in zip(axes2, metrics, ylabels):
    delta = np.array([metric[c][1] - metric[c][0] for c in categories])

    # colour-scale centred on zero for each panel
    norm = TwoSlopeNorm(vmin=-abs(delta).max(),
                        vcenter=0,
                        vmax= abs(delta).max())
    bar_colors = delta_cmap(norm(delta))

    ax.bar(x, delta, width*1.5, color=bar_colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel(f"Δ {ylabel}\n(EXP 2 – EXP 1)", fontsize=14, linespacing=1.4)
    ax.tick_params(labelsize=12)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

fig2.tight_layout()
plt.show()
