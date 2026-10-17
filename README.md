# 📊 AutoEDA Studio

**AutoEDA Studio** is an automated, modular 8-stage Exploratory Data Analysis (EDA) web application built with **Streamlit**, **Pandas**, **Scikit-Learn**, **Matplotlib**, and **Seaborn**.

It features an intelligent **SHA-256 disk caching engine** that saves all metrics as JSON and all visualizations as high-resolution PNG plots. When the same dataset is uploaded again, results are loaded instantly without recomputation.

---

## 📁 Repository Structure

```
EDA/
├── app.py              # Streamlit interactive frontend & dashboard
├── eda_pipeline.py     # Standalone 8-stage EDA pipeline & plot generator
├── eda_results/        # Auto-generated cache directory
│   └── <dataset>_<hash>/
│       ├── results.json
│       └── plots/
│           ├── 01_stat.png
│           ├── 02_missing.png
│           ├── 03_noisy.png
│           ├── 04_outliers.png
│           ├── 05_balanced.png
│           ├── 06_correlation.png
│           ├── 07_features_importance.png
│           └── 08_kde.png
└── README.md           # Documentation
```

---

## 🔬 The 8-Stage EDA Pipeline

The pipeline is structured into 8 distinct analytical stages, each producing quantitative metrics and an accompanying publication-quality visualization:

| # | Module | What It Computes | Visualization |
|---|---|---|---|
| **1** | **Stat** | Shape, memory usage, data types, and descriptive statistics (mean, std, median, min, max, quantiles, skewness, kurtosis). | Standardized Z-Score distribution boxplots for numeric features. |
| **2** | **Missing Data** | Total nulls, overall missing rate, per-column missing percentage, and complete rows count. | Horizontal missingness bar chart (% missing per column). |
| **3** | **Noisy Data** | Exact duplicate rows, zero-variance (constant) columns, quasi-constant columns (>99% dominant), and suspicious string tokens (`?`, `null`, `n/a`, whitespace). | Data hygiene and anomaly indicator bar chart. |
| **4** | **Outliers** | Tukey's 1.5× Interquartile Range (IQR) lower/upper bounds, IQR outlier counts, and Z-score (>3σ) extreme values. | Top numeric features ranked by outlier percentage. |
| **5** | **Balanced** | Class & category balance analysis across **all** discrete/categorical columns, calculating majority-to-minority imbalance ratios and severity tags (*Balanced*, *Mild*, *Moderate*, *Severe*). | Imbalance ratio comparison bar chart + focus column class distribution donut chart. |
| **6** | **Correlation** | Pairwise Pearson correlation matrix and automated detection of highly collinear feature pairs (\|r\| ≥ 0.70). | Annotated correlation matrix heatmap with divergent colormap. |
| **7** | **Feature Importance** *(Target-dependent)* | Trains a **Random Forest** (Classifier for discrete targets, Regressor for continuous targets) and computes non-linear **Mutual Information** scores. | Grouped horizontal bar chart comparing Random Forest Gini vs. Mutual Information scores. |
| **8** | **KDE** | Kernel Density Estimation continuous probability distributions across key numeric features. | Multi-panel KDE grid (stratified by target class when classification target is chosen). |

---

## ✨ Key Features

- **⚡ SHA-256 Smart Caching**: File bytes + target column determine a unique hash. Subsequent visits or repeated uploads load cached metrics and plots instantly.
- **📁 Saved Runs History**: Browse and load previous runs directly from the sidebar without needing to re-upload the original dataset.
- **🎯 Optional Target Variable**:
  - **Unsupervised EDA**: Run on any tabular dataset without a target variable (Steps 1–4, 6, 8 execute; Step 7 is cleanly skipped, and Step 5 evaluates all categorical columns).
  - **Supervised EDA**: Select any target variable in the sidebar to activate machine-learning feature importance and target-stratified distributions.
- **📑 Multi-Format Ingestion**: Supports `.csv`, `.xlsx`, and `.xls` tabular data files, with a built-in demo dataset button.
- **💻 CLI & Programmatic API**: `eda_pipeline.py` can be imported into notebooks or executed standalone from the command line.

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure Python 3.9+ is installed. The required packages include:

```bash
pip install streamlit pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
```

### 2. Launch the Web Application

From within the `EDA` folder:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Run Pipeline Standalone via CLI

You can also run the EDA pipeline directly from your terminal:

```bash
# Unsupervised analysis
python eda_pipeline.py path/to/dataset.csv

# Supervised analysis with target column
python eda_pipeline.py path/to/dataset.csv --target "target_column"

# Force re-run bypassing cache
python eda_pipeline.py path/to/dataset.csv --target "target_column" --force
```

Results and plots will be written to `eda_results/<dataset>_<hash>/`.
