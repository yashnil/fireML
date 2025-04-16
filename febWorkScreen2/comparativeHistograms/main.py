#!/usr/bin/env python3
# ============================================================
#  Comparative Metrics Histogram (X-axis = Categories)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────
#  Metric data for Tests 1–3 across categories c0..c3
# Format: [Test1, Test2, Test3]
# ────────────────────────────────────────────────────────────
bias_data = {
    'c0': [0.05, 0.06, 0.07],
    'c1': [1.08, -0.53, -0.56],
    'c2': [0.92, -0.30, -0.36],
    'c3': [4.20, 1.67, 1.53],
}

rmse_data = {
    'c0': [7.55, 11.88, 11.88],
    'c1': [14.16, 13.47, 13.48],
    'c2': [15.12, 14.20, 14.31],
    'c3': [15.88, 13.38, 13.17],
}

r2_data = {
    'c0': [0.946, 0.866, 0.866],
    'c1': [0.789, 0.823, 0.822],
    'c2': [0.774, 0.808, 0.805],
    'c3': [0.718, 0.799, 0.805],
}

# ────────────────────────────────────────────────────────────
#  Labels and Test Settings
# ────────────────────────────────────────────────────────────
categories = ['c0', 'c1', 'c2', 'c3']
test_labels = ['Test 1', 'Test 2', 'Test 3']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # dark blue, orange, green

# ────────────────────────────────────────────────────────────
#  Plotting Function (flipped axes)
# ────────────────────────────────────────────────────────────
def plot_metric_histogram(metric_dict, title, ylabel):
    x = np.arange(len(categories))  # 0..3
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(3):  # For each test
        values = [metric_dict[cat][i] for cat in categories]
        ax.bar(x + (i - 1) * width, values, width,
               label=test_labels[i], color=colors[i], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Category")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Training Regime")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────
#  Generate All 3 Charts
# ────────────────────────────────────────────────────────────
plot_metric_histogram(bias_data, "Comparative Bias Across Categories", "Bias (Pred - Obs)")
plot_metric_histogram(rmse_data, "Comparative RMSE Across Categories", "RMSE (days)")
plot_metric_histogram(r2_data, "Comparative R² Across Categories", "R² Score")
