# Run Instructions for Experiment 1 Additional Analyses

This document provides instructions for running the additional analyses requested for Experiment 1 revisions.

## Files Created

1. **RFR/sensitivity_bins.py** - Random Forest sensitivity analysis with 0.15 and 0.35 bins
2. **XGBoost/newversion.py** - XGBoost version of Experiment 1
3. **LinearRegression/newversion.py** - Linear Regression version of Experiment 1
4. **climate_skill_analysis.py** - Climate condition skill score analysis

---

## 1. Random Forest Sensitivity Analysis (0.15 and 0.35 bins)

### Purpose
Test whether results are sensitive to the choice of 0.25 burn fraction bins by evaluating alternative thresholds (0.15 and 0.35).

### How to Run
```bash
cd /Users/yashnilmohanty/Desktop/fireML/FINAL_SCREEN/RFR
python sensitivity_bins.py
```

### Output
- **CSV file**: `/Users/yashnilmohanty/Desktop/rf_sensitivity_bins_results.csv`
  - Contains metrics (RMSE, Bias, Bias_Std, R²) for each threshold (0.15, 0.25, 0.35) and burn category (c0-c3, overall)

### What It Does
- Tests three different burn fraction bin thresholds: 0.15, 0.25, and 0.35
- For each threshold, creates 4 categories based on cumulative burn fraction
- Trains Random Forest on unburned pixels (cat 0) and evaluates across all categories
- Reports metrics in a summary table

---

## 2. XGBoost Version of Experiment 1

### Purpose
Run the exact same experiment as MLP/LSTM but using XGBoost to enable side-by-side comparison.

### How to Run
```bash
cd /Users/yashnilmohanty/Desktop/fireML/FINAL_SCREEN/XGBoost
python newversion.py
```

### Output
- Same figures as MLP/LSTM:
  - Scatter plots (train/test/all data)
  - Bias histograms
  - Density scatter plots
  - Bias maps
  - Elevation×Vegetation heatmaps and boxplots
- Runs for both unburned_max_cat=0 and unburned_max_cat=1

### What It Does
- Identical structure to MLP/LSTM experiments
- Trains on 70% of unburned pixels (cat 0)
- Evaluates across all burn categories
- Uses XGBoost with default parameters (n_estimators=100, max_depth=6)

---

## 3. Linear Regression Version of Experiment 1

### Purpose
Run the exact same experiment as MLP/LSTM but using Linear Regression to enable side-by-side comparison.

### How to Run
```bash
cd /Users/yashnilmohanty/Desktop/fireML/FINAL_SCREEN/LinearRegression
python newversion.py
```

### Output
- Same figures as MLP/LSTM:
  - Scatter plots (train/test/all data)
  - Bias histograms
  - Density scatter plots
  - Bias maps
  - Elevation×Vegetation heatmaps and boxplots
- Runs for both unburned_max_cat=0 and unburned_max_cat=1

### What It Does
- Identical structure to MLP/LSTM experiments
- Trains on 70% of unburned pixels (cat 0)
- Evaluates across all burn categories
- Uses Linear Regression with standardized features

---

## 4. Climate Condition Skill Score Analysis

### Purpose
Evaluate whether model skill scores are impacted by hydrometeorological conditions (wet/cold, wet/hot, dry/cold, dry/hot) that may impact fire.

### How to Run
```bash
cd /Users/yashnilmohanty/Desktop/fireML/FINAL_SCREEN
python climate_skill_analysis.py
```

### Output
- **CSV file 1**: `/Users/yashnilmohanty/Desktop/climate_skill_scores.csv`
  - Contains skill scores (RMSE, Bias, Bias_Std, R²) for each climate condition
  - Includes number of samples and year indices for each condition
  
- **CSV file 2**: `/Users/yashnilmohanty/Desktop/climate_condition_summary.csv`
  - Contains annual precipitation and temperature means
  - Flags for above/below median conditions

### What It Does
1. **Classifies years into 4 climate conditions**:
   - **Wet/Cold**: Above-median precipitation, below-median temperature
   - **Wet/Hot**: Above-median precipitation, above-median temperature
   - **Dry/Cold**: Below-median precipitation, below-median temperature
   - **Dry/Hot**: Below-median precipitation, above-median temperature

2. **Uses spatially averaged precipitation and temperature**:
   - Default: Uses `aorcWinterPrecipitation` (or `aorcWinterRain`) for precipitation
   - Default: Uses `aorcSpringTemperature` for temperature
   - Falls back to first available variable if defaults not found

3. **Trains Random Forest on unburned pixels** and evaluates skill across climate conditions

4. **Reports metrics per condition** to assess whether skill varies with hydrometeorological conditions

### Notes
- The script automatically detects precipitation and temperature variables in the dataset
- Uses median split to classify years (can be modified in the code if needed)
- Currently uses 0.25 threshold for burn categories (can be modified if needed)

---

## Dataset Requirements

All scripts require:
- **Dataset**: `/Users/yashnilmohanty/Desktop/final_dataset5.nc`
- **Dataset structure**: Must have `year` and `pixel` dimensions
- **Required variables**: `DSD` (target), `burn_cumsum`, `Elevation`, `VegTyp`, and AORC climate variables

---

## Dependencies

All scripts require the same packages as the main project:
- numpy, pandas, xarray
- scikit-learn
- xgboost (for XGBoost script)
- matplotlib, cartopy, geopandas (for visualization scripts)
- scipy

---

## Comparison Workflow

To compare models side-by-side:

1. Run MLP experiment:
   ```bash
   cd FINAL_SCREEN/MLP && python newversion.py
   ```

2. Run LSTM experiment:
   ```bash
   cd FINAL_SCREEN/LSTM && python newversion.py
   ```

3. Run XGBoost experiment:
   ```bash
   cd FINAL_SCREEN/XGBoost && python newversion.py
   ```

4. Run Linear Regression experiment:
   ```bash
   cd FINAL_SCREEN/LinearRegression && python newversion.py
   ```

5. Compare the generated figures and metrics across all four models

---

## Notes

- All scripts use the same random seed (42) for reproducibility
- All scripts use the same train/test split (70/30) for fair comparison
- The sensitivity analysis and climate analysis scripts output CSV files for easy comparison
- The model comparison scripts (XGBoost, Linear Regression) generate the same visualization suite as MLP/LSTM

