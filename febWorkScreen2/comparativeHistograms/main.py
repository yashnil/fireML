import matplotlib.pyplot as plt
import numpy as np

# sample data
categories  = ['c0','c1','c2','c3']
bias_data   = {'c0':[.05,.06,.07],  'c1':[1.08,-.53,-.56],  'c2':[.92,-.30,-.36],  'c3':[4.20,1.67,1.53]}
rmse_data   = {'c0':[7.55,11.88,11.88],'c1':[14.16,13.47,13.48],'c2':[15.12,14.20,14.31],'c3':[15.88,13.38,13.17]}
r2_data     = {'c0':[.946,.866,.866],'c1':[.789,.823,.822],'c2':[.774,.808,.805],'c3':[.718,.799,.805]}
test_labels = ['Experiment 1','Experiment 2','Experiment 3']
colors      = ['#1f77b4','#ff7f0e','#2ca02c']

fig, axes = plt.subplots(1, 3, figsize=(15,4), sharex=True)
fig.subplots_adjust(wspace=0.5)   # add horizontal space between panels

for ax, metric, ylabel in zip(axes, (bias_data, rmse_data, r2_data),
                              ("Bias (Pred – Obs, Days)", "RMSE (days)", "R²")):
    x = np.arange(len(categories))
    width = 0.25
    for i in range(3):
        vals = [metric[c][i] for c in categories]
        ax.bar(x + (i - 1) * width, vals, width,
               color=colors[i], alpha=0.9,
               label=test_labels[i] if ax is axes[0] else None)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# leave extra room at bottom for legend, and widen space between panels
fig.subplots_adjust(bottom=0.25, wspace=0.4)

# shared legend centered under all three panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.05),
           ncol=3,
           frameon=False,
           fontsize=14)

plt.show()
