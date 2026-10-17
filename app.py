"""
Streamlit EDA Frontend Application
===================================
A modern, responsive exploratory data analysis web application.
Reads datasets (CSV, Excel), coordinates with eda_pipeline.py, and presents
the 8 EDA modules with cached metrics and publication-quality plots.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Import pipeline functions
from eda_pipeline import compute_dataset_hash, run_eda_pipeline, RESULTS_BASE_DIR

# Set page configuration
st.set_page_config(
    page_title="AutoEDA Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    /* Global styles */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    
    /* Header card */
    .eda-header {
        background: linear-gradient(135deg, #4361EE 0%, #3A0CA3 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.25);
    }
    .eda-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 2rem;
    }
    .eda-header p {
        color: #E0E7FF;
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.825rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Cache status badges */
    .badge-cached {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-computed {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        border: 1px solid #93C5FD;
    }
    
    /* Callout notice */
    .callout-box {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .callout-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Dataset Loading Helper
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or Excel dataset into a pandas DataFrame."""
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))


# ------------------------------------------------------------------------------
# Sidebar: File Upload & Configuration
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("AutoEDA Controls")
    st.markdown("Automated 8-stage Exploratory Data Analysis & Caching Engine.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload tabular data file to analyze.",
    )

    use_demo = False
    if not uploaded_file:
        st.markdown("**— OR —**")
        use_demo = st.button("🚀 Load Heart Attack Demo Dataset", use_container_width=True)

    df: Optional[pd.DataFrame] = None
    raw_bytes: Optional[bytes] = None
    file_name: str = ""

    # Load uploaded file or demo
    if uploaded_file is not None:
        file_name = uploaded_file.name
        raw_bytes = uploaded_file.getvalue()
        try:
            df = parse_uploaded_file(raw_bytes, file_name)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif use_demo or "demo_loaded" in st.session_state:
        st.session_state["demo_loaded"] = True
        demo_path = Path("Datasets/Heart_Attack_Analysis/heart.csv")
        if not demo_path.exists():
            demo_path = Path("../Datasets/Heart_Attack_Analysis/heart.csv")
        if demo_path.exists():
            file_name = "heart.csv"
            raw_bytes = demo_path.read_bytes()
            df = pd.read_csv(demo_path)

    target_col = None
    force_rerun = False

    if df is not None:
        st.success(f"Loaded `{file_name}` ({len(df):,} rows × {len(df.columns)} cols)")

        # Target Column Selector
        columns_list = ["(None - Unsupervised EDA)"] + list(df.columns)
        default_target_idx = 0
        if "output" in df.columns:
            default_target_idx = columns_list.index("output")
        elif "target" in df.columns:
            default_target_idx = columns_list.index("target")

        chosen_target = st.selectbox(
            "Target Column (Optional)",
            options=columns_list,
            index=default_target_idx,
            help="Select the prediction target variable for supervised EDA (Balanced & Feature Importance).",
        )
        if chosen_target != "(None - Unsupervised EDA)":
            target_col = chosen_target

        force_rerun = st.checkbox("Force Re-run (Bypass Cache)", value=False)
        st.divider()

        # Cache Indicator
        if raw_bytes is not None:
            cache_hash = compute_dataset_hash(raw_bytes, target_col)
            cached_file = RESULTS_BASE_DIR / cache_hash / "results.json"
            if cached_file.exists() and not force_rerun:
                st.markdown(f'<span class="badge-cached">⚡ Cache Available: {cache_hash}</span>', unsafe_allow_html=True)
                st.caption("Results will load instantly from disk.")
            else:
                st.markdown(f'<span class="badge-computed">⚙️ Fresh Run Required</span>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Main Application Content
# ------------------------------------------------------------------------------
if df is None:
    # Empty State Welcome Screen
    st.markdown("""
    <div class="eda-header">
        <h1>📊 Welcome to AutoEDA Studio</h1>
        <p>Comprehensive, structured 8-stage Exploratory Data Analysis pipeline with disk caching and publication-quality plots.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("👈 Upload a dataset in the sidebar (CSV or Excel) or click **Load Heart Attack Demo Dataset** to begin!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. General & Distributions")
        st.markdown("- **1. Stat:** Metrics, summary statistics, standardized boxplots\n- **8. KDE:** Kernel Density Estimation distributions")
    with col2:
        st.markdown("### 2. Quality & Anomalies")
        st.markdown("- **2. Missing Data:** Null rates & completeness chart\n- **3. Noisy Data:** Duplicates, constant columns, junk strings\n- **4. Outliers:** 1.5× IQR & Z-score bounds")
    with col3:
        st.markdown("### 3. Relationships & Target")
        st.markdown("- **5. Balanced:** Class imbalance ratio & severity (optional target)\n- **6. Correlation:** Pearson heatmap & collinear pairs\n- **7. Feature Importance:** Random Forest & Mutual Info (optional target)")

    st.stop()


# ------------------------------------------------------------------------------
# Run Pipeline / Load Cache
# ------------------------------------------------------------------------------
with st.spinner("Analyzing dataset & preparing EDA modules..."):
    t0 = time.time()
    results, run_dir = run_eda_pipeline(
        df=df,
        target_col=target_col,
        file_bytes=raw_bytes,
        force_rerun=force_rerun,
    )
    duration = time.time() - t0

# Header Title Banner
st.markdown(f"""
<div class="eda-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>📊 AutoEDA Studio — {file_name}</h1>
            <p>Target: <strong>{target_col if target_col else 'None (Unsupervised)'}</strong> | Cache ID: <code>{results.get('cache_key', 'unknown')}</code></p>
        </div>
        <div>
            {'<span class="badge-cached">⚡ Loaded from Cache (' + f'{duration*1000:.1f}ms)</span>' if results.get('_cached') else '<span class="badge-computed">🚀 Newly Computed (' + f'{duration:.2f}s)</span>'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Executive Metrics Overview
stat_data = results.get("1_stat", {}).get("overview", {})
missing_data = results.get("2_missing", {})
noisy_data = results.get("3_noisy", {})

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stat_data.get("n_rows", len(df)):,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stat_data.get("n_cols", len(df.columns))}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stat_data.get("n_numeric", 0)}</div><div class="metric-label">Numeric Cols</div></div>', unsafe_allow_html=True)
with m4:
    missing_pct = missing_data.get("overall_missing_pct", 0.0)
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#E71D36" if missing_pct > 0 else "#2EC4B6"}">{missing_pct}%</div><div class="metric-label">Missing Rate</div></div>', unsafe_allow_html=True)
with m5:
    dups = noisy_data.get("duplicate_rows", 0)
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#E71D36" if dups > 0 else "#2EC4B6"}">{dups}</div><div class="metric-label">Duplicates</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stat_data.get("memory_mb", 0.0)} MB</div><div class="metric-label">Memory</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 8 Structured EDA Tabs
# ------------------------------------------------------------------------------
tab_labels = [
    "1. 📋 Stat",
    "2. 🔍 Missing Data",
    "3. 🧹 Noisy Data",
    "4. 🎯 Outliers",
    "5. ⚖️ Balanced",
    "6. 🔗 Correlation",
    "7. 🌲 Feature Importance",
    "8. 📈 KDE",
]

t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(tab_labels)


# ------------------------------------------------------------------------------
# TAB 1: STAT
# ------------------------------------------------------------------------------
with t1:
    st.subheader("1. Descriptive Statistics & Distributions")
    st.markdown("Descriptive summary for all numeric variables alongside their standardized spread.")

    c1, c2 = st.columns([3, 2])
    with c1:
        num_stats = results.get("1_stat", {}).get("numeric_stats", {})
        if num_stats:
            stats_df = pd.DataFrame(num_stats).T
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True, height=350)
        else:
            st.info("No numeric columns detected in this dataset.")

    with c2:
        plot_p = run_dir / results.get("1_stat", {}).get("plot", "")
        if plot_p.exists():
            st.image(str(plot_p), caption="Standardized Distribution (Z-Scores) of Top Features", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: MISSING DATA
# ------------------------------------------------------------------------------
with t2:
    st.subheader("2. Missing Data Analysis")
    missing_info = results.get("2_missing", {})
    col_missing = missing_info.get("columns_with_missing", {})

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
        - **Total Missing Cells:** `{missing_info.get('total_nulls', 0):,}`
        - **Overall Missing Percentage:** `{missing_info.get('overall_missing_pct', 0.0)}%`
        - **Complete Rows:** `{missing_info.get('complete_rows', 0):,}` (`{missing_info.get('complete_rows_pct', 0.0)}%`)
        """)
        if col_missing:
            m_df = pd.DataFrame(col_missing).T.rename(columns={"count": "Missing Count", "percentage": "Missing %"})
            st.dataframe(m_df.style.format({"Missing Count": "{:,.0f}", "Missing %": "{:.2f}%"}), use_container_width=True)
        else:
            st.markdown('<div class="callout-box"><strong>Clean Dataset:</strong> No missing values found across all columns!</div>', unsafe_allow_html=True)

    with c2:
        plot_p = run_dir / missing_info.get("plot", "")
        if plot_p.exists():
            st.image(str(plot_p), caption="Missing Data Breakdown by Feature", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: NOISY DATA
# ------------------------------------------------------------------------------
with t3:
    st.subheader("3. Noisy & Anomalous Data Quality")
    st.markdown("Identification of duplicate rows, constant/quasi-constant columns, and junk string tokens.")

    noisy_info = results.get("3_noisy", {})
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
        - **Duplicate Rows:** `{noisy_info.get('duplicate_rows', 0)}` (`{noisy_info.get('duplicate_pct', 0.0)}%`)
        - **Constant Columns (0 variance):** `{noisy_info.get('constant_columns', [])}`
        - **Quasi-Constant Columns (>99% dominant):** `{len(noisy_info.get('quasi_constant_columns', []))}`
        """)

        placeholders = noisy_info.get("placeholder_anomalies", {})
        if placeholders:
            st.markdown("**Suspicious String Tokens Detected (`?`, `null`, `n/a`, whitespace):**")
            st.json(placeholders)

        quasi = noisy_info.get("quasi_constant_columns", [])
        if quasi:
            st.markdown("**Quasi-constant Breakdown:**")
            st.dataframe(pd.DataFrame(quasi), use_container_width=True)

    with c2:
        plot_p = run_dir / noisy_info.get("plot", "")
        if plot_p.exists():
            st.image(str(plot_p), caption="Noise & Data Hygiene Overview", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: OUTLIERS
# ------------------------------------------------------------------------------
with t4:
    st.subheader("4. Outlier Analysis (IQR & Z-Score)")
    st.markdown("Evaluation of extreme values using Tukey's 1.5× Interquartile Range and standard Z-score (|z| > 3).")

    outlier_info = results.get("4_outliers", {})
    details = outlier_info.get("outlier_details", {})

    c1, c2 = st.columns([3, 2])
    with c1:
        if details:
            outlier_df = pd.DataFrame(details).T.rename(columns={
                "iqr_outliers": "IQR Count",
                "iqr_percentage": "IQR %",
                "lower_bound": "Lower Threshold",
                "upper_bound": "Upper Threshold",
                "zscore_outliers": "Z-Score Count (>3σ)"
            })
            st.dataframe(outlier_df.style.format({
                "IQR Count": "{:,.0f}",
                "IQR %": "{:.2f}%",
                "Lower Threshold": "{:.2f}",
                "Upper Threshold": "{:.2f}",
                "Z-Score Count (>3σ)": "{:,.0f}"
            }), use_container_width=True, height=350)
        else:
            st.info("No numeric features suitable for outlier computation.")

    with c2:
        plot_p = run_dir / outlier_info.get("plot", "")
        if plot_p.exists():
            st.image(str(plot_p), caption="Top Outlier Percentages (1.5× IQR)", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 5: BALANCED (Classification)
# ------------------------------------------------------------------------------
with t5:
    st.subheader("5. Target Class Balance Assessment")
    balance_info = results.get("5_balanced", {})

    if balance_info.get("status") == "skipped":
        st.markdown(f"""
        <div class="callout-warning">
            <strong>Target-Dependent Section Skipped:</strong> {balance_info.get('reason')}.
            <br><em>To enable class balance analysis, pick a categorical or discrete classification target column in the sidebar.</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
            - **Target Column:** `{balance_info.get('target_column')}`
            - **Number of Classes:** `{balance_info.get('n_classes')}`
            - **Imbalance Ratio (Majority / Minority):** `{balance_info.get('imbalance_ratio')}:1`
            - **Status:** **{balance_info.get('severity')}**
            """)

            classes_df = pd.DataFrame({
                "Count": balance_info.get("class_counts", {}),
                "Percentage (%)": balance_info.get("class_percentages", {})
            })
            st.dataframe(classes_df.style.format({"Count": "{:,.0f}", "Percentage (%)": "{:.2f}%"}), use_container_width=True)

        with c2:
            plot_p = run_dir / balance_info.get("plot", "")
            if plot_p.exists():
                st.image(str(plot_p), caption=f"Class Balance Distribution ({balance_info.get('severity')})", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 6: CORRELATION
# ------------------------------------------------------------------------------
with t6:
    st.subheader("6. Feature Correlation & Collinearity")
    st.markdown("Pairwise Pearson correlations and detection of multicollinear feature pairs (|r| ≥ 0.70).")

    corr_info = results.get("6_correlation", {})
    if corr_info.get("status") == "skipped":
        st.info(corr_info.get("reason"))
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            high_pairs = corr_info.get("high_correlation_pairs", [])
            if high_pairs:
                st.markdown(f"**Highly Correlated Pairs Found (`{len(high_pairs)}`):**")
                p_df = pd.DataFrame(high_pairs)[["feature_1", "feature_2", "correlation"]]
                st.dataframe(p_df.style.format({"correlation": "{:.3f}"}), use_container_width=True)
            else:
                st.markdown('<div class="callout-box">No pairs exceed the collinearity threshold of |r| ≥ 0.70.</div>', unsafe_allow_html=True)

        with c2:
            plot_p = run_dir / corr_info.get("plot", "")
            if plot_p.exists():
                st.image(str(plot_p), caption="Correlation Matrix Heatmap", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 7: FEATURES IMPORTANCE
# ------------------------------------------------------------------------------
with t7:
    st.subheader("7. Machine Learning Feature Importance")
    st.markdown("Feature importance rankings computed via a trained Random Forest model and non-linear Mutual Information.")

    fi_info = results.get("7_features_importance", {})
    if fi_info.get("status") == "skipped":
        st.markdown(f"""
        <div class="callout-warning">
            <strong>Target-Dependent Section Skipped:</strong> {fi_info.get('reason')}.
            <br><em>To train a Random Forest model and calculate feature importances, select a target variable in the sidebar.</em>
        </div>
        """, unsafe_allow_html=True)
    elif fi_info.get("status") == "error":
        st.error(f"Error computing feature importance: {fi_info.get('error')}")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
            - **Target Column:** `{fi_info.get('target')}`
            - **Model Type:** `Random Forest {fi_info.get('task_type')}` + `Mutual Information`
            """)
            top_f = fi_info.get("top_features", [])
            if top_f:
                fi_df = pd.DataFrame(top_f).rename(columns={
                    "feature": "Feature",
                    "rf_importance": "RF Importance",
                    "mi_score": "Mutual Information"
                })
                st.dataframe(fi_df.style.format({
                    "RF Importance": "{:.4f}",
                    "Mutual Information": "{:.4f}"
                }), use_container_width=True)

        with c2:
            plot_p = run_dir / fi_info.get("plot", "")
            if plot_p.exists():
                st.image(str(plot_p), caption=f"Top Feature Importance Scores ({fi_info.get('task_type')})", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 8: KDE
# ------------------------------------------------------------------------------
with t8:
    st.subheader("8. Kernel Density Estimation (KDE)")
    st.markdown("Continuous probability density distributions for top numeric features (grouped by target class if selected).")

    kde_info = results.get("8_kde", {})
    if kde_info.get("status") == "skipped":
        st.info(kde_info.get("reason"))
    else:
        st.markdown(f"- **Features Plotted:** `{', '.join(kde_info.get('features_plotted', []))}`")
        if kde_info.get("split_by_target"):
            st.markdown(f"- **Stratified by Target Class:** Yes (`{target_col}`)")

        plot_p = run_dir / kde_info.get("plot", "")
        if plot_p.exists():
            st.image(str(plot_p), caption="Kernel Density Estimation (KDE) Grid", use_container_width=True)
