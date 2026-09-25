# Machine Learning Insights for Snow Disappearance Predictability in Northern California Burned Areas

This repository contains the data processing, model training, and evaluation code for a study of how wildfire affects the predictability of the Day of Snow Disappearance (DSD) across Northern California, using machine learning models trained on satellite-derived snow observations and gridded hydrometeorological, topographic, and land-cover predictors.

## Problem Description

Seasonal snowpack in the mountains of Northern California is a primary source of water for agriculture, municipal supply, and hydropower in California. The timing of snow disappearance controls when meltwater enters streams and reservoirs, so accurate DSD prediction is important for water resource planning.

Wildfire is increasingly altering these mountain landscapes. Removing forest canopy changes how much solar radiation reaches the snow surface, how much snow is intercepted by vegetation, and how dark the snow surface becomes due to deposited soot and debris. These processes can shift snow disappearance earlier in burned areas. Statistical and machine learning models that are trained mostly on unburned terrain may therefore learn predictor–DSD relationships that no longer hold after a fire.

This project addresses three questions:

1. How well do models trained only on unburned pixels predict DSD in areas with increasing cumulative burn history?
2. Does including burned pixels in the training data (implicit representation of fire) improve predictions in burned areas?
3. Does adding explicit fire and vegetation information as predictors (burn fraction, FPAR) provide further improvement?

## Study Domain and Data

The study covers Northern California over the years 2004–2018. Each sample is a single pixel in a single year; static predictors (topography, land cover) are repeated across years.

| Dataset | Variables used |
|:--|:--|
| **MODIS MOD10A2** (via the ORNL snow disappearance product) | Target variable: Day of Snow Disappearance (DSD) |
| **SNODAS** | Peak snow water equivalent (SWE), winter mean SWE, date of peak SWE |
| **AORC** | Seasonal (fall, winter, spring) means of temperature, precipitation, shortwave radiation, longwave radiation, and humidity |
| **MTBS** | Annual burned-area fraction per pixel, accumulated into a cumulative burn fraction |
| **SRTM** | Elevation, slope, aspect |
| **GLCC** (USGS) | Vegetation / land-cover type |
| **MODIS FPAR** | Fraction of Absorbed Photosynthetically Active Radiation (Experiment 4 only) |

Preprocessing notes:

- Summer AORC variables are excluded from the predictor set, since they occur after snow disappearance.
- Winter precipitation is converted from mm s<sup>-1</sup> to mm day<sup>-1</sup>.
- Annual burn fractions are capped at 1 and accumulated over time (including burns from 2001–2003) to produce the cumulative burn fraction `burn_cumsum`.
- Samples with missing predictor or target values are dropped.

The processed NetCDF datasets (`final_dataset4.nc`, `final_dataset5.nc`, `final_dataset6.nc`) are not included in this repository because of their size.

## Methodology

### Burn categories

Each pixel-year is assigned to one of four categories based on its cumulative burn fraction (BF):

| Category | Cumulative burn fraction |
|:--:|:--|
| c0 (unburned) | BF < 0.25 |
| c1 | 0.25 ≤ BF < 0.50 |
| c2 | 0.50 ≤ BF < 0.75 |
| c3 | BF ≥ 0.75 |

A sensitivity analysis repeats the main experiment with alternative bin thresholds of 0.15 and 0.35 to check that conclusions do not depend on the 0.25 choice.

### Models

| Model | Configuration |
|:--|:--|
| Random Forest (primary model) | scikit-learn `RandomForestRegressor`, 100 trees |
| Multilayer Perceptron | scikit-learn `MLPRegressor`, two hidden layers of 64 units, Adam optimizer, standardized inputs |
| Long Short-Term Memory | Keras, one LSTM layer of 32 units and a linear output layer, 100 epochs, batch size 512, standardized inputs and target |
| XGBoost | `XGBRegressor`, 100 estimators, max depth 6, learning rate 0.1 |
| Linear Regression | Ordinary least squares on principal components that retain 95% of the variance of the standardized predictors, to reduce multicollinearity |

All experiments use a 70% / 30% train / test split and a fixed random seed (42) for reproducibility.

### Experiments

| Experiment | Training data | Predictors | Purpose |
|:--:|:--|:--|:--|
| **1** | 70% of unburned (c0) pixels | Base predictors (no burn fraction) | Establish a baseline trained on unburned conditions and measure how skill degrades when applied to burned categories c1–c3 |
| **2** | 70% of each burn category (c0–c3), stratified | Base predictors (no burn fraction) | Test whether implicitly including burned samples in training improves predictions in burned areas |
| **3** | Same as Experiment 2 | Base predictors + burn fraction | Test whether explicitly providing fire history as a predictor adds skill |
| **4** | Same as Experiment 2 | Base predictors + FPAR | Test whether dynamic vegetation information (e.g., post-fire regrowth) adds skill |

Experiment 1 is run with all five model types to confirm that its conclusions are not specific to Random Forests. Experiments 2–4 use Random Forests.

### Supplementary analyses

- **Controlled comparison** (`IDENTICAL/main.py`): evaluates the Experiment 1 and Experiment 3 models on the same held-out test set so their skill can be compared directly.
- **Per-category training** (`per_category_experiment.py`, `robustness.py`): trains and tests a separate model within each burn category.
- **Burned-only training** (`rf_burned_stratified.py`): trains on burned categories (c1–c3) only.
- **Climate conditions** (`climate_skill_analysis.py`): classifies years as wet/cold, wet/hot, dry/cold, or dry/hot using median splits of winter precipitation and spring temperature, and reports skill for each group.
- **Extreme dry winters**: reports skill separately for the driest 5% of samples (by a snowfall-frequency proxy) and for the remaining 95%.
- **Burned-area trend** (`burn_trend.py`): tests for a significant increase in annual burned area in Northern California with a one-tailed linear regression.

### Evaluation

Model skill is reported overall and for each burn category using:

- Coefficient of determination (R²)
- Root mean squared error (RMSE)
- Mean bias (predicted minus observed DSD)
- Standard deviation of bias

Differences in bias distributions between burn categories are tested with the Wilcoxon rank-sum test. Monotonic relationships between individual predictors and DSD are measured with Spearman rank correlation. Predictor importance is assessed with mean decrease in impurity (MDI) and permutation importance. Bias is also mapped spatially and summarized across elevation and vegetation-type bins.

## Key Findings

- Models trained only on unburned pixels systematically overestimate snow duration (predict DSD too late) in heavily burned pixels, and skill decreases as cumulative burn fraction increases.
- Including burned pixels in the training data improves predictions in burned areas, particularly in the most severely burned category.
- Adding burn fraction or FPAR as explicit predictors provides only marginal additional improvement; implicit representation through training samples is more effective.
- Prediction bias varies with elevation, vegetation type, and burn severity, and this spatial heterogeneity persists after fire.
- Peak SWE, spring temperature, and elevation are the most important predictors; burn fraction ranks low in importance.

## Repository Structure

```
fireML/
├── preprocessing/              Build meteorological, fire, and static datasets from raw sources
├── modelTraining/              Merge datasets, check alignment, handle missing values, correlation analysis
├── INITIAL_SCREEN/             Early exploratory analysis
├── FINAL_SCREEN/               Experiments and analyses reported in the paper
│   ├── RFR/                    Experiment 1: Random Forest (plus bin-threshold sensitivity analysis)
│   ├── MLP/                    Experiment 1: Multilayer Perceptron
│   ├── LSTM/                   Experiment 1: LSTM
│   ├── XGBoost/                Experiment 1: XGBoost
│   ├── LinearRegression/       Experiment 1: PCA + Linear Regression
│   ├── excludeBurnFraction/    Experiment 2: stratified training without burn fraction
│   ├── includeBurnFraction/    Experiment 3: stratified training with burn fraction
│   ├── experiment4/            Experiment 4: stratified training with FPAR
│   ├── IDENTICAL/              Controlled comparison of Experiments 1 and 3 on a shared test set
│   ├── comparativeHistograms/  Metric comparison plots and feature / permutation importance
│   ├── climate_skill_analysis.py
│   ├── per_category_experiment.py
│   ├── rf_burned_stratified.py
│   ├── robustness.py
│   ├── burn_trend.py
│   └── RUN_INSTRUCTIONS.md     Detailed instructions for the supplementary analyses
├── data/                       U.S. state boundary shapefile used for maps
└── requirements.txt
```

In each model directory, `newversion.py` is the current experiment script. Files prefixed with `backedup` or `backed` are earlier versions kept for reference.

## Getting Started

### Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies: NumPy, pandas, SciPy, scikit-learn, XGBoost, TensorFlow/Keras, xarray, netCDF4, rasterio, GeoPandas, Cartopy, Matplotlib, and seaborn.

### Data paths

The scripts read the processed NetCDF datasets from absolute paths (for example, `xr.open_dataset("/Users/.../final_dataset5.nc")`). Update these paths to point to your local copy of the data before running.

### Running experiments

From the repository root:

```bash
# Experiment 1 (replace RFR with MLP, LSTM, XGBoost, or LinearRegression)
python FINAL_SCREEN/RFR/newversion.py

# Experiments 2 and 3
python FINAL_SCREEN/excludeBurnFraction/newversion.py
python FINAL_SCREEN/includeBurnFraction/newversion.py

# Experiment 4
python FINAL_SCREEN/experiment4/new.py
```

### Generating comparison plots

```bash
python FINAL_SCREEN/comparativeHistograms/main.py
python FINAL_SCREEN/comparativeHistograms/featureImportance.py
python FINAL_SCREEN/comparativeHistograms/permutationImportance.py
```

See `FINAL_SCREEN/RUN_INSTRUCTIONS.md` for the sensitivity, XGBoost, Linear Regression, and climate-condition analyses.

## Citation

If you use this work, please cite:

> Mohanty, Y., Abolafia-Rosenzweig, R., He, C., & McGrath, D. (2025). *Machine Learning Insights for Snow Disappearance Predictability in Northern California Burned Areas*. Environmental Research Letters (in review).

## Future Work

- Hybrid models that combine machine learning with physically based snow models
- Additional fire-sensitive predictors, such as soil burn severity and surface albedo
- Extension of the approach to other snow-dominated regions

## Contact

- Yashnil Mohanty: yashnilmohanty@gmail.com (GitHub: [yashnil](https://github.com/yashnil))
- R. Abolafia-Rosenzweig: abolafia@ucar.edu

## License

This project is licensed under the MIT License.
