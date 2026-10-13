"""
app.py
──────
Salary Predictor — Streamlit Web Application

Usage:
    streamlit run app.py
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn
import streamlit as st

# scikit-learn 1.6+ compatibility
sklearn.set_config(transform_output="default")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global typography */
html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", sans-serif; }

/* Header banner */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #3b82d4 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { font-size: 2.4rem; margin: 0; font-weight: 700; }
.main-header p  { font-size: 1.05rem; margin: 0.4rem 0 0; opacity: 0.88; }

/* Metric cards */
.metric-card {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label { font-size: 0.85rem; color: #57606a; margin-bottom: 0.2rem; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; }

/* Prediction result box */
.result-box {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f0fe 100%);
    border: 2px solid #3b82d4;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-box .salary { font-size: 2.8rem; font-weight: 800; color: #1e3a5f; }
.result-box .range  { font-size: 1rem; color: #57606a; margin-top: 0.3rem; }

/* Disclaimer */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #5d4037;
    margin-top: 1rem;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 2px solid #3b82d4;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH    = "models/best_model.joblib"
METADATA_PATH = "models/metadata.joblib"
DATA_PATH     = "data/salary_data.csv"
CHART_DIR     = "charts"


@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Load the trained pipeline and metadata; auto-train if files are missing."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        with st.spinner("🔧 First run: training the model — this takes ~30 seconds …"):
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error("Model training failed. See terminal output.")
                st.code(result.stderr)
                st.stop()
    model    = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


model, metadata = load_model_and_metadata()
edu_options = metadata["edu_categories"]
job_options = metadata["job_categories"]
loc_options = metadata["loc_categories"]
best_model  = metadata["best_model"]
results     = metadata["results"]
df_raw      = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>💼 Salary Predictor</h1>
    <p>Estimate your annual salary using Machine Learning — powered by real career data patterns</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict Salary", "📊 Data Insights", "🤖 Model Performance"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_form, col_info = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Enter Your Details</div>', unsafe_allow_html=True)

        education  = st.selectbox("🎓 Education Level",  edu_options, index=2)
        experience = st.number_input("📅 Years of Experience", min_value=0.0,
                                     max_value=45.0, value=5.0, step=0.5,
                                     help="Total professional work experience in years")
        job_title  = st.selectbox("💼 Job Title",         job_options)
        location   = st.selectbox("📍 Location",          loc_options)
        age        = st.number_input("🎂 Age",             min_value=18, max_value=70,
                                     value=28, step=1,
                                     help="Your current age in years")

        # Basic input validation
        min_age = int(experience) + 18
        if age < min_age:
            st.warning(f"⚠️ Age ({age}) seems low for {int(experience)} years of experience. "
                       f"Consider age ≥ {min_age}.")

        predict_btn = st.button("🚀 Predict Salary", type="primary", use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
The model analyses five factors to estimate your annual compensation:

| Factor | Impact |
|---|---|
| 🎓 Education Level | Higher degree → higher base salary |
| 📅 Years of Experience | Each year adds ~$2,200 on average |
| 💼 Job Title | Technical roles command a premium |
| 📍 Location | Cost-of-living adjustments apply |
| 🎂 Age | Used as a proxy for career stage |

**Model used:** """ + best_model + """

The prediction is a statistical estimate based on market patterns in the training dataset.
        """)

        if predict_btn:
            input_df = pd.DataFrame([{
                "age":        float(age),
                "education":  education,
                "job_title":  job_title,
                "location":   location,
                "experience": float(experience),
            }])
            predicted = model.predict(input_df)[0]
            low  = predicted * 0.90
            high = predicted * 1.10

            st.markdown(f"""
<div class="result-box">
    <div style="font-size:1rem; color:#57606a; margin-bottom:0.5rem;">Estimated Annual Salary</div>
    <div class="salary">${predicted:,.0f}</div>
    <div class="range">Likely range: ${low:,.0f} — ${high:,.0f}</div>
</div>
""", unsafe_allow_html=True)

    # Disclaimer always visible
    st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This prediction is a <em>statistical estimate</em> produced by a
    machine learning model trained on synthetic data. It does <strong>not</strong> constitute a job
    offer, salary guarantee, or professional HR advice. Actual salaries vary based on company size,
    industry, negotiation, and many other factors not captured here.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset not found at `data/salary_data.csv`. Run `train_model.py` first.")
    else:
        df = df_raw.dropna()

        # ── KPI row ──────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
<div class="metric-card">
<div class="label">Total Records</div>
<div class="value">{len(df):,}</div>
</div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
<div class="metric-card">
<div class="label">Median Salary</div>
<div class="value">${df['salary'].median():,.0f}</div>
</div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
<div class="metric-card">
<div class="label">Avg Experience</div>
<div class="value">{df['experience'].mean():.1f} yrs</div>
</div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
<div class="metric-card">
<div class="label">Unique Job Titles</div>
<div class="value">{df['job_title'].nunique()}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        sns.set_theme(style="whitegrid", palette="muted")

        row1_c1, row1_c2 = st.columns(2)

        with row1_c1:
            st.markdown("**Annual Salary Distribution**")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.histplot(df["salary"], bins=35, kde=True, color="#3b82d4", ax=ax)
            ax.set_xlabel("Annual Salary (USD)")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Salaries follow a roughly normal distribution skewed right for senior technical roles.")

        with row1_c2:
            st.markdown("**Salary by Education Level**")
            edu_order = ["High School", "Associate", "Bachelor", "Master", "PhD"]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.boxplot(data=df, x="education", y="salary", order=edu_order,
                        palette="Blues", ax=ax)
            ax.set_xlabel("Education")
            ax.set_ylabel("Annual Salary (USD)")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Each education tier shows a clear median salary step up, most pronounced at PhD level.")

        row2_c1, row2_c2 = st.columns(2)

        with row2_c1:
            st.markdown("**Salary vs Years of Experience**")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.scatterplot(data=df, x="experience", y="salary", alpha=0.3,
                            color="#7c5cd8", ax=ax)
            sns.regplot(data=df, x="experience", y="salary",
                        scatter=False, color="red", ax=ax)
            ax.set_xlabel("Years of Experience")
            ax.set_ylabel("Annual Salary (USD)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            corr = df["experience"].corr(df["salary"])
            st.caption(f"Positive linear relationship (Pearson r = {corr:.3f}). More experience → higher pay.")

        with row2_c2:
            st.markdown("**Median Salary by Job Title**")
            med_job = df.groupby("job_title")["salary"].median().sort_values()
            fig, ax = plt.subplots(figsize=(7, 3.8))
            med_job.plot(kind="barh", color="#3b82d4", ax=ax)
            ax.set_xlabel("Median Salary (USD)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Top role: {med_job.idxmax()}  |  Lowest: {med_job.idxmin()}")

        st.markdown("**Median Salary by Location**")
        med_loc = df.groupby("location")["salary"].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        med_loc.plot(kind="bar", color="#7c5cd8", ax=ax)
        ax.set_xlabel("Location")
        ax.set_ylabel("Median Salary (USD)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"San Francisco and New York lead; Dallas and Miami are lowest due to cost-of-living adjustments.")

        # ── Correlation heatmap ───────────────────────────────────────────────
        st.markdown("**Feature Correlation Heatmap**")
        num_df = df[["age", "experience", "salary"]].copy()
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    linewidths=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Experience has the strongest correlation with salary among numerical features.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Model Evaluation Results</div>', unsafe_allow_html=True)

    st.markdown("Two models were trained on 80% of the dataset and evaluated on the held-out 20%.")

    # ── Metrics table ─────────────────────────────────────────────────────────
    rows = []
    for mname, mres in results.items():
        rows.append({
            "Model":    mname,
            "MAE ($)":  f"${mres['MAE']:,.2f}",
            "RMSE ($)": f"${mres['RMSE']:,.2f}",
            "R² Score": f"{mres['R2']:.4f}",
            "Best ✅":   "✅" if mname == best_model else "",
        })
    st.table(pd.DataFrame(rows).set_index("Model"))

    # ── Bar chart comparison ──────────────────────────────────────────────────
    model_names = list(results.keys())
    mae_vals    = [results[m]["MAE"]  for m in model_names]
    rmse_vals   = [results[m]["RMSE"] for m in model_names]
    r2_vals     = [results[m]["R2"]   for m in model_names]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**MAE & RMSE Comparison**")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w/2, mae_vals,  w, label="MAE",  color="#3b82d4")
        ax.bar(x + w/2, rmse_vals, w, label="RMSE", color="#7c5cd8")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=10)
        ax.set_ylabel("Error (USD)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**R² Score Comparison**")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#3b82d4" if m != best_model else "#22863a" for m in model_names]
        ax.bar(model_names, r2_vals, color=colors)
        ax.set_ylabel("R² Score (higher = better)")
        ax.set_ylim(0, 1)
        for i, v in enumerate(r2_vals):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Residual chart if saved ───────────────────────────────────────────────
    residual_path = f"{CHART_DIR}/residuals_{best_model.replace(' ', '_')}.png"
    if os.path.exists(residual_path):
        st.markdown(f"**Residual Analysis — {best_model}**")
        st.image(residual_path, use_container_width=True)
        st.caption("Left: residuals centered around zero = unbiased model. "
                   "Right: points close to the diagonal = accurate predictions.")

    st.markdown("""
---
**Interpretation Guide**
- **MAE** — Mean Absolute Error: the average dollar amount the prediction is off by.
- **RMSE** — Root Mean Squared Error: penalises large errors more heavily than MAE.
- **R²** — Coefficient of determination: proportion of salary variance explained by the model (1.0 = perfect).
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#57606a; font-size:0.82rem;'>"
    "💼 Salary Predictor · Built with Python & Streamlit · "
    "Predictions are estimates only and do not constitute salary offers."
    "</center>",
    unsafe_allow_html=True,
)
