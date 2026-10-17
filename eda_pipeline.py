"""
EDA Pipeline Module
===================
Handles the 8-stage Exploratory Data Analysis and Matplotlib/Seaborn plot generation:
1. Stat (Descriptive statistics & distributions)
2. Missing Data (Null analysis & missingness rates)
3. Noisy Data (Duplicates, zero-variance, junk/placeholders)
4. Outliers (IQR and Z-Score outlier detection)
5. Balanced (Target classification distribution & imbalance ratio)
6. Correlation (Pearson/Spearman matrix & collinear pairs)
7. Features Importance (Random Forest & Mutual Information)
8. KDE (Kernel Density Estimation distributions)

Results are cached to disk as results.json and static PNG plots keyed by content SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/streamlit use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

# Base directory for EDA results
EDA_DIR = Path(__file__).resolve().parent
RESULTS_BASE_DIR = EDA_DIR / "eda_results"

# Visual Theme Configuration
PALETTE_PRIMARY = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0", "#4895EF"]
COLOR_PRIMARY = "#4361EE"
COLOR_ACCENT = "#F72585"
COLOR_SUCCESS = "#2EC4B6"
COLOR_WARNING = "#FF9F1C"
COLOR_DANGER = "#E71D36"


def set_plot_style():
    """Apply modern, publication-quality aesthetic styling for all plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
    })


def compute_dataset_hash(file_bytes: bytes, target_col: Optional[str] = None) -> str:
    """Compute a SHA-256 hash from file content bytes and target column."""
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    if target_col:
        hasher.update(f"__target:{target_col}".encode("utf-8"))
    return hasher.hexdigest()[:16]


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python types for valid JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif pd.isna(obj):
        return None
    return obj


# ==============================================================================
# 1. STAT
# ==============================================================================
def run_stat(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """1. Descriptive Statistics & Summary Distributions."""
    set_plot_style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    overview = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    num_stats = {}
    for col in num_cols:
        s = df[col].dropna()
        if len(s) > 0:
            num_stats[col] = {
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
                "median": round(float(s.median()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "q25": round(float(s.quantile(0.25)), 3),
                "q75": round(float(s.quantile(0.75)), 3),
                "skewness": round(float(s.skew()), 3) if len(s) > 2 else 0.0,
                "kurtosis": round(float(s.kurtosis()), 3) if len(s) > 3 else 0.0,
            }

    # Plot generation
    plot_rel_path = "plots/01_stat.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    if num_cols:
        # Standardize numeric columns to compare distributions on same scale
        plot_cols = num_cols[:8]  # Limit to first 8 for visual clarity
        std_data = []
        labels = []
        for col in plot_cols:
            s = df[col].dropna()
            if s.std() > 0:
                std_data.append((s - s.mean()) / s.std())
                labels.append(col)
        
        if std_data:
            box = ax.boxplot(std_data, tick_labels=labels, patch_artist=True,
                             medianprops=dict(color="#E71D36", linewidth=1.5),
                             boxprops=dict(facecolor="#4361EE", alpha=0.6, color="#3A0CA3"))
            ax.set_title("Standardized Distributions of Numeric Features (Z-Scores)", pad=12)
            ax.set_ylabel("Standardized Value (Z-Score)")
            plt.xticks(rotation=25, ha="right")
        else:
            ax.text(0.5, 0.5, "No numeric variation to display", ha="center", va="center")
    else:
        # If no numeric columns, plot categorical unique counts
        uniques = [df[c].nunique() for c in cat_cols[:8]]
        ax.bar(cat_cols[:8], uniques, color=COLOR_PRIMARY, alpha=0.8)
        ax.set_title("Unique Value Counts of Categorical Features", pad=12)
        ax.set_ylabel("Number of Unique Categories")
        plt.xticks(rotation=25, ha="right")

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "overview": overview,
        "numeric_stats": num_stats,
        "plot": plot_rel_path,
    }


# ==============================================================================
# 2. MISSING DATA
# ==============================================================================
def run_missing(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """2. Missing Data Analysis & Rates."""
    set_plot_style()
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    total_cells = df.size
    total_nulls = int(null_counts.sum())
    overall_missing_pct = round((total_nulls / total_cells * 100), 2) if total_cells > 0 else 0.0
    complete_rows = int(df.dropna().shape[0])
    complete_rows_pct = round((complete_rows / len(df) * 100), 2) if len(df) > 0 else 0.0

    missing_cols_info = {}
    for col in df.columns:
        cnt = int(null_counts[col])
        if cnt > 0:
            missing_cols_info[col] = {
                "count": cnt,
                "percentage": float(null_pct[col]),
            }

    # Plot generation
    plot_rel_path = "plots/02_missing.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sorted_null = null_pct[null_pct > 0].sort_values(ascending=True)

    if len(sorted_null) > 0:
        bars = ax.barh(sorted_null.index.astype(str), sorted_null.values, color=COLOR_WARNING, alpha=0.85)
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_title("Missing Data Percentage by Feature", pad=12)
        ax.set_xlim(0, max(sorted_null.max() * 1.15, 10))
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                    va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax.bar(["No Missing Data"], [100], color=COLOR_SUCCESS, alpha=0.8, width=0.4)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Completeness (%)")
        ax.set_title("Dataset Completeness: 100% Clean (No Missing Values)", pad=12)
        ax.text(0, 50, "Perfect Completeness!\nZero NaN / Null Values Detected",
                ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "total_nulls": total_nulls,
        "overall_missing_pct": overall_missing_pct,
        "complete_rows": complete_rows,
        "complete_rows_pct": complete_rows_pct,
        "columns_with_missing": missing_cols_info,
        "plot": plot_rel_path,
    }


# ==============================================================================
# 3. NOISY DATA
# ==============================================================================
def run_noisy(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """3. Noisy Data Detection: Duplicates, Constant Features, Placeholders."""
    set_plot_style()
    n_rows = len(df)
    n_dups = int(df.duplicated().sum())
    dup_pct = round((n_dups / n_rows * 100), 2) if n_rows > 0 else 0.0

    # 1. Constant & Quasi-constant (>=99% same value)
    constant_cols = []
    quasi_constant_cols = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            top_freq_pct = (s.value_counts(normalize=True).iloc[0] * 100)
            if s.nunique() == 1:
                constant_cols.append(col)
            elif top_freq_pct >= 99.0 and s.nunique() > 1:
                quasi_constant_cols.append({"column": col, "dominant_pct": round(top_freq_pct, 2)})

    # 2. Suspicious string placeholders in object/string columns
    SUSPICIOUS_TOKENS = {"?", "none", "null", "n/a", "na", "-", "--", "missing", "unknown", ""}
    placeholder_counts = {}
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        s_str = df[col].astype(str).str.strip().str.lower()
        cnt = int(s_str.isin(SUSPICIOUS_TOKENS).sum())
        if cnt > 0:
            placeholder_counts[col] = cnt

    # Plot generation
    plot_rel_path = "plots/03_noisy.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    categories = ["Duplicate Rows", "Constant Columns", "Quasi-Const (>99%)", "Cols w/ Junk/Tokens"]
    counts = [n_dups, len(constant_cols), len(quasi_constant_cols), len(placeholder_counts)]
    colors = [COLOR_DANGER if c > 0 else COLOR_SUCCESS for c in counts]

    bars = ax.bar(categories, counts, color=colors, alpha=0.85, width=0.55)
    ax.set_title("Noisy & Anomaly Data Quality Indicators", pad=12)
    ax.set_ylabel("Count Detected")
    ax.set_ylim(0, max(max(counts) * 1.25, 5))

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, str(int(h)),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "duplicate_rows": n_dups,
        "duplicate_pct": dup_pct,
        "constant_columns": constant_cols,
        "quasi_constant_columns": quasi_constant_cols,
        "placeholder_anomalies": placeholder_counts,
        "plot": plot_rel_path,
    }


# ==============================================================================
# 4. OUTLIERS
# ==============================================================================
def run_outliers(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """4. Outlier Analysis via IQR & Z-Score for Numeric Features."""
    set_plot_style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_results = {}
    iqr_counts = {}

    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 5 or s.nunique() <= 2:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        n_iqr = int(((s < lower_iqr) | (s > upper_iqr)).sum())
        iqr_pct = round((n_iqr / len(s) * 100), 2)

        # Z-score (|z| > 3)
        std_val = s.std()
        if std_val > 0:
            z_scores = np.abs((s - s.mean()) / std_val)
            n_zscore = int((z_scores > 3.0).sum())
        else:
            n_zscore = 0

        outlier_results[col] = {
            "iqr_outliers": n_iqr,
            "iqr_percentage": iqr_pct,
            "lower_bound": round(lower_iqr, 2),
            "upper_bound": round(upper_iqr, 2),
            "zscore_outliers": n_zscore,
        }
        iqr_counts[col] = iqr_pct

    # Plot generation
    plot_rel_path = "plots/04_outliers.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sorted_iqr = pd.Series(iqr_counts).sort_values(ascending=False)

    if len(sorted_iqr) > 0:
        top_iqr = sorted_iqr.head(10)
        bars = ax.barh(top_iqr.index.astype(str), top_iqr.values, color=COLOR_ACCENT, alpha=0.85)
        ax.set_xlabel("Outlier Percentage (% of column via 1.5x IQR)")
        ax.set_title("Top Numeric Features by Outlier Percentage (1.5x IQR Rule)", pad=12)
        ax.invert_yaxis()
        ax.set_xlim(0, max(top_iqr.max() * 1.2, 5))
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                    va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No numeric columns with outliers detected", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "features_analyzed": len(outlier_results),
        "outlier_details": outlier_results,
        "plot": plot_rel_path,
    }


# ==============================================================================
# 5. BALANCED (Classification & Discrete Features)
# ==============================================================================
def run_balanced(df: pd.DataFrame, target_col: Optional[str], out_dir: Path) -> Dict[str, Any]:
    """5. Class & Category Balance Assessment for All Discrete/Categorical Columns."""
    set_plot_style()

    # Identify all discrete or categorical columns (<= 30 unique values or non-numeric)
    candidate_cols = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 0 and (s.nunique() <= 30 or not is_numeric_dtype(s.dtype)):
            candidate_cols.append(col)

    if not candidate_cols:
        return {
            "status": "skipped",
            "reason": "No discrete or categorical columns found (all columns appear continuous with high cardinality).",
        }

    columns_results: Dict[str, Any] = {}
    summary_list: List[Dict[str, Any]] = []

    for col in candidate_cols:
        s = df[col].dropna()
        n_unique = int(s.nunique())
        val_counts = s.value_counts()
        percentages = (s.value_counts(normalize=True) * 100).round(2)
        maj_cnt = int(val_counts.iloc[0])
        min_cnt = int(val_counts.iloc[-1])
        ratio = round(maj_cnt / max(min_cnt, 1), 2)

        if ratio <= 1.5:
            severity = "Balanced"
        elif ratio <= 3.0:
            severity = "Mild Imbalance"
        elif ratio <= 10.0:
            severity = "Moderate Imbalance"
        else:
            severity = "Severe Imbalance"

        col_dict = {
            "column": col,
            "n_classes": n_unique,
            "majority_class": str(val_counts.index[0]),
            "majority_count": maj_cnt,
            "minority_class": str(val_counts.index[-1]),
            "minority_count": min_cnt,
            "imbalance_ratio": ratio,
            "severity": severity,
            "class_counts": {str(k): int(v) for k, v in val_counts.items()},
            "class_percentages": {str(k): float(v) for k, v in percentages.items()},
        }
        columns_results[col] = col_dict
        summary_list.append(col_dict)

    # Sort summary list by highest imbalance ratio
    summary_list = sorted(summary_list, key=lambda x: x["imbalance_ratio"], reverse=True)

    # Plot generation
    plot_rel_path = "plots/05_balanced.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    # Pick focus column: target_col if eligible, otherwise highest imbalanced column
    focus_col = None
    if target_col and target_col in columns_results:
        focus_col = target_col
    else:
        focus_col = summary_list[0]["column"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # Left plot: All columns ranked by Imbalance Ratio
    plot_summary = summary_list[:15]  # Limit to top 15 columns for visual cleanliness
    col_names = [x["column"] for x in plot_summary]
    ratios = [x["imbalance_ratio"] for x in plot_summary]
    bar_colors = [
        COLOR_SUCCESS if r <= 1.5 else
        COLOR_WARNING if r <= 3.0 else
        "#FB8500" if r <= 10.0 else COLOR_DANGER
        for r in ratios
    ]

    bars = ax1.barh(col_names, ratios, color=bar_colors, alpha=0.85)
    ax1.axvline(1.5, color="#2EC4B6", linestyle="--", linewidth=1.5, label="Balanced Threshold (1.5x)")
    ax1.axvline(3.0, color="#FF9F1C", linestyle=":", linewidth=1.2, label="Mild / Moderate (3.0x)")
    ax1.invert_yaxis()
    ax1.set_xlabel("Imbalance Ratio (Majority / Minority)")
    ax1.set_title("Category Imbalance Ratio Across Columns", pad=12)
    ax1.legend(loc="lower right", fontsize=8)

    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.1f}x",
                 va="center", ha="left", fontsize=9, fontweight="bold")

    # Right plot: Detailed focus column distribution
    focus_info = columns_results[focus_col]
    focus_counts = focus_info["class_counts"]
    f_labels = [str(k) for k in focus_counts.keys()][:8]
    f_vals = [focus_counts[k] for k in f_labels]

    ax2.pie(f_vals, labels=f_labels, autopct="%1.1f%%",
            startangle=140, colors=PALETTE_PRIMARY[:len(f_labels)],
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
    focus_title_suffix = " (Selected Target)" if focus_col == target_col else ""
    ax2.set_title(f"Focus: '{focus_col}'{focus_title_suffix}\n({focus_info['severity']} - Ratio {focus_info['imbalance_ratio']}x)", pad=12)

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "completed",
        "target_column": target_col,
        "focus_column": focus_col,
        "total_columns_evaluated": len(columns_results),
        "columns": columns_results,
        "summary_list": summary_list,
        "plot": plot_rel_path,
    }



# ==============================================================================
# 6. CORRELATION
# ==============================================================================
def run_correlation(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """6. Multicollinearity & Correlation Analysis."""
    set_plot_style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        return {"status": "skipped", "reason": "Fewer than 2 numeric columns available for correlation."}

    corr_matrix = df[num_cols].corr(method="pearson").round(3)

    # Find top correlated pairs (|r| > 0.70)
    high_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1 = num_cols[i]
            col2 = num_cols[j]
            r = corr_matrix.loc[col1, col2]
            if not np.isnan(r) and abs(r) >= 0.70:
                high_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(float(r), 3),
                    "abs_correlation": round(abs(float(r)), 3)
                })

    high_pairs = sorted(high_pairs, key=lambda x: x["abs_correlation"], reverse=True)

    # Plot generation
    plot_rel_path = "plots/06_correlation.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    # Limit heatmap size if too many columns
    cols_to_plot = num_cols[:15]
    sub_corr = df[cols_to_plot].corr()

    fig_size = max(7, min(14, len(cols_to_plot) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85), dpi=150)
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    
    sns.heatmap(sub_corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                annot=len(cols_to_plot) <= 10, fmt=".2f",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap (Pearson)", pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "completed",
        "numeric_features": len(num_cols),
        "high_correlation_pairs": high_pairs[:15],
        "plot": plot_rel_path,
    }


# ==============================================================================
# 7. FEATURES IMPORTANCE (Target-dependent)
# ==============================================================================
def run_features_importance(df: pd.DataFrame, target_col: Optional[str], out_dir: Path) -> Dict[str, Any]:
    """7. Feature Importance via Random Forest and Mutual Information."""
    if not target_col or target_col not in df.columns:
        return {"status": "skipped", "reason": "No target column selected"}

    set_plot_style()
    df_clean = df.dropna(subset=[target_col]).copy()
    if len(df_clean) < 10:
        return {"status": "skipped", "reason": "Insufficient samples (< 10) to train feature importance model."}

    y_raw = df_clean[target_col]
    X_raw = df_clean.drop(columns=[target_col])

    if X_raw.empty:
        return {"status": "skipped", "reason": "No predictors available apart from target column."}

    # Encode categorical features for modeling
    X_encoded = pd.DataFrame(index=X_raw.index)
    for col in X_raw.columns:
        if is_numeric_dtype(X_raw[col].dtype):
            X_encoded[col] = X_raw[col].fillna(X_raw[col].median())
        else:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_encoded[col] = oe.fit_transform(X_raw[[col]].astype(str).fillna("missing")).ravel()

    # Determine problem type
    is_classification = y_raw.nunique() <= 20 or not is_numeric_dtype(y_raw.dtype)

    try:
        if is_classification:
            y_encoded = pd.factorize(y_raw)[0]
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_encoded, y_encoded)
            rf_importances = model.feature_importances_
            mi_scores = mutual_info_classif(X_encoded, y_encoded, random_state=42)
            task_name = "Classification"
        else:
            y_num = y_raw.astype(float)
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_encoded, y_num)
            rf_importances = model.feature_importances_
            mi_scores = mutual_info_regression(X_encoded, y_num, random_state=42)
            task_name = "Regression"

        # Normalize MI scores to sum to 1 for comparison
        mi_sum = mi_scores.sum()
        mi_norm = (mi_scores / mi_sum) if mi_sum > 0 else mi_scores

        fi_df = pd.DataFrame({
            "feature": X_encoded.columns,
            "rf_importance": rf_importances,
            "mi_score": mi_scores,
            "mi_normalized": mi_norm
        }).sort_values(by="rf_importance", ascending=False)

        top_fi = fi_df.head(12)

        # Plot generation
        plot_rel_path = "plots/07_features_importance.png"
        plot_file = out_dir / plot_rel_path
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        y_pos = np.arange(len(top_fi))
        bar_height = 0.38

        ax.barh(y_pos - bar_height/2, top_fi["rf_importance"], height=bar_height,
                color=COLOR_PRIMARY, label="Random Forest (Gini/Variance)", alpha=0.85)
        ax.barh(y_pos + bar_height/2, top_fi["mi_normalized"], height=bar_height,
                color=COLOR_SUCCESS, label="Mutual Information (Normalized)", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_fi["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Relative Importance Score")
        ax.set_title(f"Top Feature Importances for Target '{target_col}' ({task_name})", pad=12)
        ax.legend(loc="lower right")

        plt.tight_layout()
        fig.savefig(plot_file, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "completed",
            "task_type": task_name,
            "target": target_col,
            "top_features": [
                {
                    "feature": str(row["feature"]),
                    "rf_importance": round(float(row["rf_importance"]), 4),
                    "mi_score": round(float(row["mi_score"]), 4),
                }
                for _, row in top_fi.iterrows()
            ],
            "plot": plot_rel_path,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==============================================================================
# 8. KDE (Kernel Density Estimation)
# ==============================================================================
def run_kde(df: pd.DataFrame, target_col: Optional[str], out_dir: Path) -> Dict[str, Any]:
    """8. Kernel Density Estimation (KDE) Distribution Plots."""
    set_plot_style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]

    if not num_cols:
        return {"status": "skipped", "reason": "No continuous numeric features available for KDE."}

    # Pick up to top 6 numeric features with variance
    valid_cols = [c for c in num_cols if df[c].dropna().std() > 0][:6]
    if not valid_cols:
        return {"status": "skipped", "reason": "No numeric columns with variance > 0."}

    # Determine if we can split KDE by target class
    can_split_target = False
    if target_col and target_col in df.columns:
        if df[target_col].nunique() <= 5:
            can_split_target = True

    plot_rel_path = "plots/08_kde.png"
    plot_file = out_dir / plot_rel_path
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    n_plots = len(valid_cols)
    n_cols_grid = min(3, n_plots)
    n_rows_grid = math.ceil(n_plots / n_cols_grid)

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows_grid), dpi=150)
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(valid_cols):
        ax = axes[idx]
        if can_split_target:
            for cls_val in df[target_col].dropna().unique()[:4]:
                subset = df[df[target_col] == cls_val][col].dropna()
                if len(subset) > 3 and subset.std() > 0:
                    sns.kdeplot(subset, ax=ax, label=f"{target_col}={cls_val}", fill=True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        else:
            subset = df[col].dropna()
            sns.kdeplot(subset, ax=ax, color=COLOR_PRIMARY, fill=True, alpha=0.35, linewidth=2)

        ax.set_title(f"KDE: {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density")
        ax.set_xlabel(col)

    # Hide any unused subplots in the grid
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "completed",
        "features_plotted": valid_cols,
        "split_by_target": can_split_target,
        "plot": plot_rel_path,
    }


# ==============================================================================
# MASTER PIPELINE ORCHESTRATOR & CACHING
# ==============================================================================
def run_eda_pipeline(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    dataset_name: str = "dataset",
    file_bytes: Optional[bytes] = None,
    force_rerun: bool = False,
    results_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Run or load the 8-stage EDA pipeline.
    If results exist in cache and force_rerun is False, loads instantly from disk.
    """
    import re
    from datetime import datetime

    if results_dir is None:
        results_dir = RESULTS_BASE_DIR

    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate cache key
    if file_bytes is not None:
        cache_key = compute_dataset_hash(file_bytes, target_col)
    else:
        # Fallback hash based on df structure & target
        content_repr = f"{df.shape}_{list(df.columns)}_{df.iloc[:5].to_dict()}_{target_col}".encode("utf-8")
        cache_key = hashlib.sha256(content_repr).hexdigest()[:16]

    clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_name).strip("_") or "dataset"
    expected_folder_name = f"{clean_name}_{cache_key}"

    # Check if an existing run matching this cache_key exists (either clean_name_key or raw key)
    run_dir = None
    if not force_rerun:
        if (results_dir / expected_folder_name / "results.json").exists():
            run_dir = results_dir / expected_folder_name
        elif (results_dir / cache_key / "results.json").exists():
            run_dir = results_dir / cache_key
        else:
            # Check any folder ending with cache_key
            for match in results_dir.glob(f"*_{cache_key}/results.json"):
                run_dir = match.parent
                break

    if run_dir is not None and not force_rerun:
        json_path = run_dir / "results.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_data["_cached"] = True
            # Backfill dataset_name if missing in older cache
            if not cached_data.get("dataset_name"):
                cached_data["dataset_name"] = dataset_name
            return cached_data, run_dir
        except Exception:
            pass

    # Destination directory for fresh run
    run_dir = results_dir / expected_folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "results.json"

    # Recompute all 8 steps
    results: Dict[str, Any] = {
        "cache_key": cache_key,
        "dataset_name": dataset_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_col": target_col,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "_cached": False,
        "1_stat": run_stat(df, run_dir),
        "2_missing": run_missing(df, run_dir),
        "3_noisy": run_noisy(df, run_dir),
        "4_outliers": run_outliers(df, run_dir),
        "5_balanced": run_balanced(df, target_col, run_dir),
        "6_correlation": run_correlation(df, run_dir),
        "7_features_importance": run_features_importance(df, target_col, run_dir),
        "8_kde": run_kde(df, target_col, run_dir),
    }

    sanitized = sanitize_for_json(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    return sanitized, run_dir


def list_saved_runs(results_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan results directory and return all saved EDA runs sorted by timestamp (newest first)."""
    from datetime import datetime

    if results_dir is None:
        results_dir = RESULTS_BASE_DIR

    if not results_dir.exists():
        return []

    runs = []
    # Search all results.json files in subdirectories
    for json_path in results_dir.glob("*/results.json"):
        try:
            stat = json_path.stat()
            mtime = stat.st_mtime
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            folder_name = json_path.parent.name
            dataset_name = data.get("dataset_name")
            if not dataset_name:
                # Infer from folder name or column features
                if "_" in folder_name:
                    dataset_name = folder_name.rsplit("_", 1)[0]
                elif "cp" in str(data.get("1_stat", {})):
                    dataset_name = "heart.csv"
                else:
                    dataset_name = f"dataset_{folder_name[:8]}"

            created_at = data.get("created_at")
            if not created_at:
                created_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            target = data.get("target_col")
            n_rows = data.get("n_rows") or data.get("1_stat", {}).get("overview", {}).get("n_rows", 0)
            n_cols = data.get("n_cols") or data.get("1_stat", {}).get("overview", {}).get("n_cols", 0)

            runs.append({
                "json_path": str(json_path),
                "run_dir": str(json_path.parent),
                "folder_name": folder_name,
                "dataset_name": dataset_name,
                "target_col": target,
                "created_at": created_at,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "cache_key": data.get("cache_key", folder_name),
                "mtime": mtime,
            })
        except Exception:
            continue

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def load_saved_run(json_path_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Load a specific saved run by its results.json path."""
    p = Path(json_path_str)
    if not p.exists():
        return None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_cached"] = True
        return data, p.parent
    except Exception:
        return None, None



# ==============================================================================
# CLI EXECUTION HELPER
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 8-stage EDA pipeline standalone.")
    parser.add_argument("file", help="Path to CSV or Excel file")
    parser.add_argument("--target", default=None, help="Optional target column name")
    parser.add_argument("--force", action="store_true", help="Force re-run and bypass cache")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        exit(1)

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df_in = pd.read_excel(file_path)
    else:
        df_in = pd.read_csv(file_path)

    print(f"Executing EDA Pipeline on '{file_path.name}' (Shape: {df_in.shape})...")
    res, out_path = run_eda_pipeline(df_in, target_col=args.target, file_bytes=raw_bytes, force_rerun=args.force)
    print(f"EDA successfully completed! Results saved to: {out_path}")
