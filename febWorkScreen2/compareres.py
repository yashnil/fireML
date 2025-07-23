import pandas as pd
from scipy.stats import ranksums

# Base paths to the run1 and run2 bias CSV directories
run1_dir = "/Users/yashnilmohanty/Desktop/run1_bias_csv"
run2_dir = "/Users/yashnilmohanty/Desktop/run2_bias_csv"

# Loop over categories 0–3
for c in range(4):
    # Construct file paths
    file1 = f"{run1_dir}/run1_bias_c{c}.csv"
    file2 = f"{run2_dir}/run2_bias_c{c}.csv"

    # Load the bias values
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    data1 = df1["bias_days"].dropna().values
    data2 = df2["bias_days"].dropna().values

    # Perform two‑sided Wilcoxon rank‑sum test
    stat, p_value = ranksums(data1, data2)

    # Report
    print(f"Category {c}:")
    print(f"  N₁ = {len(data1)}, N₂ = {len(data2)}")
    print(f"  W‑statistic = {stat:.3f}")
    print(f"  two‑sided p‑value = {p_value:.3g}\n")
