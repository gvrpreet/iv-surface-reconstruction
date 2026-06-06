# NIFTY 50 Implied Volatility Surface Reconstruction

A quantitative framework for reconstructing sparse and noisy NIFTY 50 implied volatility surfaces using spline-based structural fitting and residual gradient boosting.

The system combines:

* cubic smoothing splines,
* analytical surface derivatives,
* temporal memory features,
* spatial strike local features,
* residual XGBoost correction,
* purged time series validation.

The reconstruction pipeline operates in the total variance domain and is designed to stabilize interpolation behavior while preserving localized intraday market structure.

---

# Quantitative Motivation

Directly modeling implied volatility surfaces is unstable in sparse intraday option chains, particularly near expiry where:

* implied volatility becomes numerically unstable,
* liquidity deteriorates,
* strike coverage becomes inconsistent,
* local smile geometry becomes noisy.

To address this, the framework separates the reconstruction problem into:

1. Structural surface estimation
2. Localized residual correction

The spline layer captures the dominant smile geometry, while the machine learning layer focuses only on localized deviations from the structural baseline.

---

# Mathematical Representation

## Total Variance Domain

Instead of modeling implied volatility directly, the system operates in the total variance domain:

```text
w = σ² × τ
```

where:

| Symbol | Meaning            |
| ------ | ------------------ |
| `w`    | Total variance     |
| `σ`    | Implied volatility |
| `τ`    | Time to expiry     |

This transformation improves:

* numerical conditioning,
* cross strike smoothness,
* interpolation stability,
* residual learning behavior near expiry.

---

## Surface Coordinates

The volatility surface is parameterized using:

* log moneyness,
* time to expiry.

Log moneyness is defined as:

```text
k = ln(S / K)
```

where:

| Symbol | Meaning          |
| ------ | ---------------- |
| `S`    | Underlying price |
| `K`    | Strike price     |
| `k`    | Log moneyness    |

Using log moneyness instead of raw strike values produces smoother smile geometry and more stable spline fitting.

---

# Reconstruction Pipeline

```text
Raw Option Chain
        │
        ▼
Contract Parsing & Reshaping
        │
        ▼
Total Variance Transformation
        │
        ▼
Spline Surface Baseline
        │
        ├── Local Surface Derivatives
        │
        ▼
Residual Construction
        │
        ▼
Localized XGBoost Correction
        │
        ▼
Expiry Damping
        │
        ▼
Final IV Surface Reconstruction
```

The architecture separates:

* low frequency structural geometry,
* high frequency microstructure deviations.

This improves robustness in sparse regions while preserving local smile dynamics.

---

# Methodology

## 1. Structural Surface Baseline

For each timestamp:

* total variance is fitted against log moneyness,
* cubic smoothing splines are used to reconstruct the baseline smile.

The spline baseline acts as the structural prior for the volatility surface.

### Boundary Extrapolation

Deep OTM regions are handled using strict flat extrapolation:

* wing variance is clamped,
* first derivatives are forced to zero,
* second derivatives are forced to zero.

This prevents unstable extrapolation outside observed strike ranges.

---

## 2. Analytical Surface Derivatives

The spline framework additionally extracts:

* local skew,
* local convexity.

### Local Skew

```text
∂w/∂k
```

Measures local smile slope across strikes.

### Local Convexity

```text
∂²w/∂k²
```

Measures local smile curvature and butterfly structure.

These derivative features inject local surface geometry directly into the residual learner.

The model therefore learns:

* smile slope,
* curvature behavior,
* strike local structure,

instead of operating as a purely tabular regressor.

---

# Residual Learning Framework

The machine learning model predicts residual variance instead of raw implied volatility:

```text
w_residual = w_true − w_baseline
```

This decomposition separates:

* global surface geometry,
* localized intraday distortions.

The residual learner focuses specifically on:

* liquidity imbalances,
* local smile irregularities,
* strike level distortions,
* transient market microstructure effects.

---

# Feature Engineering

## Temporal Memory Features

Historical variance memory is constructed independently for each:

* strike,
* option type.

Generated lag features include:

```text
w_lag_1
w_lag_3
w_lag_6
```

These capture short term persistence in the volatility surface.

---

## Spatial Strike Features

Cross strike neighborhood information is injected using:

* upper adjacent strike variance,
* lower adjacent strike variance.

This allows the model to learn:

* local strike continuity,
* smile neighborhood relationships,
* surface smoothness across strikes.

---

## Market State Features

Additional state aware features include:

* time to expiry,
* time to close,
* day of week.

These features help the model adapt to intraday market regimes.

---

# Expiry Damping

Residual predictions are damped as expiry approaches:

```text
w_final = w_baseline + λ(τ) × w_residual
```

As expiry approaches:

```text
τ → 0
```

the residual component is progressively forced toward zero.

This stabilizes predictions in ultra short dated regions where:

* gamma effects dominate,
* spreads widen,
* implied volatility estimates become noisy.

The damping mechanism prevents residual amplification near expiration and anchors the reconstruction to the structural spline baseline.

---

# Validation Framework

The project uses a Purged Group Time Series Split with embargo windows.

Validation windows are constructed using:

* rolling training periods,
* explicit purge gaps,
* forward validation segments.

This prevents:

* temporal leakage,
* overlapping market state contamination,
* intraday autocorrelation leakage.

The validation scheme better reflects real world forward inference conditions compared to naive random splits.

---

# Repository Structure

```text
IV-SURFACE-NIFTY/

├── assets/

├── data/

├── notebooks/
│   ├── experiments/
│   │   ├── basic_cubic_splines.ipynb
│   │   └── lgbm_taylor_weights.ipynb
│   │
│   ├── eda_and_splines.ipynb
│   └── master_pipeline.ipynb

├── src/
│   ├── __init__.py
│   ├── data_engineering.py
│   ├── models.py
│   └── splines.py

├── .gitignore
├── README.md
└── requirements.txt
```

---

# Module Responsibilities

## `src/data_engineering.py`

Responsible for:

* option chain reshaping,
* contract parsing,
* expiry extraction,
* total variance transformation,
* log moneyness construction,
* lag feature generation,
* strike neighborhood feature generation,
* missing value handling,
* purged time series split construction.

### Core Transformations

#### Total Variance

```python
df_long['w'] = (df_long['iv'] ** 2) * df_long['tau']
```

#### Log Moneyness

```python
df_long['log_moneyness'] = np.log(
    df_long['underlying_price'] / df_long['strike']
)
```

---

## `src/splines.py`

Implements:

* cubic smoothing spline fitting,
* baseline variance reconstruction,
* strict flat wing extrapolation,
* analytical derivative extraction.

### Generated Outputs

```text
w_baseline
dw_dk
d2w_dk2
```

The spline layer provides:

* structural smile geometry,
* local slope information,
* curvature information.

---

## `src/models.py`

Implements:

* residual target generation,
* XGBoost training,
* fold aggregation,
* expiry damping,
* final IV reconstruction.

### Model Features

```python
[
    'log_moneyness',
    'tau',
    'w_lag_1',
    'w_lag_3',
    'w_lag_6',
    'w_strike_up',
    'w_strike_down',
    'dw_dk',
    'd2w_dk2',
    'time_to_close',
    'day_of_week'
]
```

---

# XGBoost Configuration

```python
{
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 1
}
```

The model is intentionally constrained to reduce overfitting on sparse intraday option surfaces.

---

# Experimental Diagnostics

The repository includes experimental notebooks evaluating:

* spline smoothness behavior,
* interpolation stability,
* LightGBM vs XGBoost behavior,
* Taylor weighted boosting approaches,
* residual learning effectiveness,
* near expiry instability,
* leakage effects in naive validation schemes.

Rejected approaches are retained for:

* reproducibility,
* research traceability,
* comparative diagnostics.

---

# Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Place raw dataset files inside:

```text
data/
```

Run the primary execution notebook:

```text
notebooks/master_pipeline.ipynb
```

Global seeds are fixed for deterministic reproducibility:

```python
RANDOM_SEED = 1
```

---

# Notes

The framework focuses on:

* volatility surface reconstruction,
* interpolation stability,
* microstructure aware feature engineering,
* constrained residual learning,
* robust forward validation.

The project is designed for sparse and noisy intraday option chain environments where direct implied volatility prediction becomes unstable.

---

Developed for the IIT Roorkee Finance Club Open Projects 2026.
