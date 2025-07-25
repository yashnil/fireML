import pandas as pd
from scipy.stats import ranksums

# Base paths to the bias CSV directories
run1_dir = "/Users/yashnilmohanty/Desktop/run1_bias_csv"
run2_dir = "/Users/yashnilmohanty/Desktop/run2_bias_csv"
run3_dir = "/Users/yashnilmohanty/Desktop/run3_bias_csv"

# Loop over categories 0–3
for c in range(4):
    print(f"Category {c}:")
    # Construct file paths
    file1 = f"{run1_dir}/run1_bias_c{c}.csv"
    file2 = f"{run2_dir}/run2_bias_c{c}.csv"
    file3 = f"{run3_dir}/run3_bias_c{c}.csv"

    # Load the bias values
    data1 = pd.read_csv(file1)["bias_days"].dropna().values
    data2 = pd.read_csv(file2)["bias_days"].dropna().values
    data3 = pd.read_csv(file3)["bias_days"].dropna().values

    # 1 vs 2
    stat12, p12 = ranksums(data1, data2)
    print(f"  Exp1 vs Exp2: N₁={len(data1)}, N₂={len(data2)}, W={stat12:.3f}, p={p12:.3g}")

    # 2 vs 3
    stat23, p23 = ranksums(data2, data3)
    print(f"  Exp2 vs Exp3: N₂={len(data2)}, N₃={len(data3)}, W={stat23:.3f}, p={p23:.3g}\n")
