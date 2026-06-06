# NIFTY 50 Implied Volatility Surface Reconstruction

## 1. Project Overview

This repository provides a production-grade quantitative pipeline for NIFTY 50 implied volatility (IV) surface reconstruction. The system addresses data sparsity and microstructure noise by utilizing a hybrid approach: **Parametric Macro-Spline baselining** coupled with **localized Gradient Boosted Decision Trees (XGBoost).**

## 2. Quantitative Philosophy: Anchor & Scalpel

The architecture deconstructs surface reconstruction into two distinct phases to separate global structural geometry from local tactical noise.

### Phase 1: Structural Anchoring (The Anchor)

A Smoothing Spline baseline forces the reconstruction to exist within the realm of physical possibility.

* **Financial Guardianship:** This baseline enforces smoothness and ensures no-arbitrage compliance, preventing irrational market state outputs.
* **Dimensionality Reduction:** By offloading the global volatility smile to the parametric model, the machine learning corrector focuses exclusively on local tactical adjustments.

### Phase 2: Residual Learning (The Scalpel)

The machine learning component operates exclusively on the residuals ($True - Baseline$).

* **Precision Targeting:** The model focuses predictive power entirely on the "blind spots" of the baseline—news-driven pivots, liquidity shocks, and intraday order flow imbalances.
* **Hybrid Generalization:** By layering local ML adjustments onto a foundation of known financial geometry, the engine remains stable where the market is smooth and adaptive where the market is chaotic.

## 3. Methodology & Implementation

### Mathematical Anchors

* **Domain Shift:** Models predicting raw IV often explode as time-to-maturity ($\tau$) approaches zero. The pipeline uses the **Total Variance domain** ($w = IV^2 \times \tau$) to ensure stability.
* **Boundary Extrapolation:** Spline boundaries enforce flat extrapolation, preventing arbitrage in deep OTM wings.

### Microstructure Correction

* **Spatial Windowing:** Cross-strike adjacency features respect butterfly spread parity.
* **Derivative Injection:** Analytically derived Skew ($\frac{\partial w}{\partial k}$) and Convexity ($\frac{\partial^2 w}{\partial k^2}$) features provide structural awareness to the trees.
* **Expiry Damping:** A linear multiplier forces ML residuals to zero near expiration, anchoring predictions to the parametric baseline.

### Time-Series Hygiene

The pipeline utilizes a **Purged Group Time Series Split** with a 1-day embargo period, preventing look-ahead bias and memorization of tick-level autocorrelation.

## 4. Experimental Diagnostics

The repository documents rejected hypotheses to maintain a clear research audit trail:

* **LightGBM/Hessian Traps:** Failure analysis of native weighting mechanisms versus XGBoost’s `hist` tree method.
* **Regime Switching Failures:** Diagnostic evidence demonstrating why multi-model bifurcation was redundant for single-expiry datasets.

## 5. Technical Architecture

```text
nifty-iv-surface/
├── assets/                  # Mathematical visualizations
├── data/                    # (Git-ignored) Raw dataset storage
├── notebooks/                      
│   ├── experiments/         # Diagnostics of rejected methodologies
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   └── 02_pipeline.ipynb    # Final production execution script
├── src/                     # Modularized quantitative engine
│   ├── data_engineering.py  # Spatial windowing and memory logic
│   ├── splines.py           # Macro-smile mathematical baselines
│   └── models.py            # Damped XGBoost corrector & boundary damping
├── requirements.txt         # Environment reproducibility lock
└── README.md

```

## 6. Execution

1. **Environment:** Install dependencies via `pip install -r requirements.txt`.
2. **Data:** Populate `data/raw/` with the project dataset.
3. **Execution:** Run `notebooks/02_pipeline.ipynb`. Global seeds (`RANDOM_SEED = 1`) ensure 100% computational reproducibility.

---
This repository structure adheres to professional quantitative engineering standards, ensuring modularity, reproducibility, and a clear research audit trail.

## 7. File Structure Overview

Your directory is organized to separate the **execution pipeline** from **exploratory research** and **mathematical logic**.

* **`assets/`**: Dedicated storage for visual outputs (charts, volatility surface plots) to be embedded in your documentation.
* **`data/raw/`**: Contains the source CSV files. *Note: The `.gitignore` prevents these from being committed to GitHub.*
* **`notebooks/`**: Serves as the user interface for your research.
* `experiments/`: A "graveyard" for discarded hypotheses. Essential for proving you performed rigorous testing.
* `eda_and_splines.ipynb`: Documenting your initial analysis and spline verification.
* `master_pipeline.ipynb`: The primary executable script that ties all modules together for submission.


* **`src/`**: The core quantitative engine. By moving logic here, your code becomes unit-testable and version-controlled.
* **`.gitignore`**: Essential for preventing large binary files or local environment variables from leaking into the repository.
* **`requirements.txt`**: Pins your library versions to ensure that any reviewer can replicate your exact environment.

## 8. Module Responsibilities

This decomposition is designed for maintainability:

| File | Responsibility |
| --- | --- |
| **`data_engineering.py`** | Handles raw-to-feature transformation, spatial/temporal lag generation, and missing value imputation. Ensures no data leakage occurs during feature construction. |
| **`splines.py`** | Houses the mathematical "Anchor." Contains the logic for fitting cubic splines to the Total Variance domain ($w = IV^2 \times \tau$) and extracting analytical derivatives. |
| **`models.py`** | Contains the machine learning "Scalpel." Executes the damped XGBoost training loop, residual calculation, and final inferential logic. |



*Developed for the IIT Roorkee Finance Club Quantitative Challenge.*