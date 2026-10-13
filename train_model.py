"""
train_model.py
──────────────
Salary Predictor – Model Training Script
Performs:
  1. Data loading & cleaning
  2. Exploratory Data Analysis (EDA) with saved charts
  3. Feature engineering & preprocessing pipeline
  4. Training Linear Regression and Random Forest models
  5. Model evaluation (MAE, RMSE, R²)
  6. Saving the best model + preprocessing pipeline via Joblib

Usage:
    python train_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

# scikit-learn 1.6+ changed internal validation — set global config to avoid errors
sklearn.set_config(transform_output="default")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/salary_data.csv"
MODEL_DIR   = "models"
CHART_DIR   = "charts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Salary Predictor — Training Script")
print("=" * 60)

# Auto-generate dataset if it doesn't exist
if not os.path.exists(DATA_PATH):
    print("\n[INFO] Dataset not found — running generate_data.py …")
    import generate_data  # noqa: F401  (executes on import)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Raw dataset shape : {df.shape}")
print(df.head())

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n    Duplicates removed  : {before - len(df)}")

# Fill numeric NaNs with median
for col in ["experience", "age"]:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    df[col] = df[col].astype(float)

# Enforce types
df["salary"]     = pd.to_numeric(df["salary"],     errors="coerce")
df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
df["age"]        = pd.to_numeric(df["age"],        errors="coerce")
df.dropna(subset=["salary"], inplace=True)

print(f"    Clean dataset shape : {df.shape}")
print("\n    Descriptive statistics (salary):")
print(df["salary"].describe().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Generating EDA charts …")

# ── 2a. Salary Distribution ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df["salary"], bins=40, kde=True, color="#3b82d4", ax=ax)
ax.set_title("Annual Salary Distribution", fontsize=14)
ax.set_xlabel("Annual Salary (USD)")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_distribution.png", dpi=120)
plt.close(fig)
print("    Saved: charts/salary_distribution.png")

# ── 2b. Salary by Education ──────────────────────────────────────────────────
edu_order = ["High School", "Associate", "Bachelor", "Master", "PhD"]
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=df, x="education", y="salary", order=edu_order,
            palette="Blues", ax=ax)
ax.set_title("Salary by Education Level", fontsize=14)
ax.set_xlabel("Education Level")
ax.set_ylabel("Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_by_education.png", dpi=120)
plt.close(fig)
print("    Saved: charts/salary_by_education.png")

# ── 2c. Salary vs Experience ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
sns.scatterplot(data=df, x="experience", y="salary", alpha=0.4,
                color="#7c5cd8", ax=ax)
sns.regplot(data=df, x="experience", y="salary", scatter=False,
            color="red", ax=ax)
ax.set_title("Salary vs Years of Experience", fontsize=14)
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_vs_experience.png", dpi=120)
plt.close(fig)
print("    Saved: charts/salary_vs_experience.png")

# ── 2d. Median Salary by Job Title ──────────────────────────────────────────
med_job = df.groupby("job_title")["salary"].median().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
med_job.plot(kind="barh", color="#3b82d4", ax=ax)
ax.set_title("Median Salary by Job Title", fontsize=14)
ax.set_xlabel("Median Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_by_job.png", dpi=120)
plt.close(fig)
print("    Saved: charts/salary_by_job.png")

# ── 2e. Median Salary by Location ────────────────────────────────────────────
med_loc = df.groupby("location")["salary"].median().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
med_loc.plot(kind="barh", color="#7c5cd8", ax=ax)
ax.set_title("Median Salary by Location", fontsize=14)
ax.set_xlabel("Median Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_by_location.png", dpi=120)
plt.close(fig)
print("    Saved: charts/salary_by_location.png")

# ── EDA observations ─────────────────────────────────────────────────────────
print("\n    Key EDA Observations:")
print(f"      • Median salary         : ${df['salary'].median():>10,.0f}")
print(f"      • Mean salary           : ${df['salary'].mean():>10,.0f}")
print(f"      • Highest median job    : {med_job.idxmax()}")
print(f"      • Lowest  median job    : {med_job.idxmin()}")
print(f"      • Highest median loc.   : {med_loc.idxmax()}")
print(f"      • Lowest  median loc.   : {med_loc.idxmin()}")
corr_exp = df["experience"].corr(df["salary"])
print(f"      • Experience↔Salary corr: {corr_exp:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building preprocessing pipeline …")

FEATURES    = ["age", "education", "job_title", "location", "experience"]
TARGET      = "salary"

X = df[FEATURES]
y = df[TARGET]

# Categorical columns and their possible ordered categories (for OrdinalEncoder)
CAT_COLS = ["education", "job_title", "location"]
NUM_COLS = ["age", "experience"]

edu_categories    = ["High School", "Associate", "Bachelor", "Master", "PhD"]
job_categories    = sorted(df["job_title"].unique().tolist())
loc_categories    = sorted(df["location"].unique().tolist())

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OrdinalEncoder(
        categories=[edu_categories, job_categories, loc_categories],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ), CAT_COLS),
], remainder="drop")

# ── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)# Force writeable copies — fixes scikit-learn 1.6 validate_data(force_writeable=True) error
X_train = X_train.copy()
X_test  = X_test.copy()
y_train = np.array(y_train, dtype=np.float64)
y_test  = np.array(y_test,  dtype=np.float64)
print(f"    Train size: {len(X_train)}  |  Test size: {len(X_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Training models …")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42,
                                               n_jobs=-1),
}

results = {}
pipelines = {}

for name, estimator in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name]   = {"MAE": mae, "RMSE": rmse, "R2": r2}
    pipelines[name] = pipe
    print(f"\n    {name}")
    print(f"      MAE  : ${mae:>10,.2f}")
    print(f"      RMSE : ${rmse:>10,.2f}")
    print(f"      R²   : {r2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESIDUAL PLOT FOR BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]["R2"])
best_pipe = pipelines[best_name]
y_pred_best = best_pipe.predict(X_test)
residuals   = y_test.values - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.4, color="#3b82d4")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title(f"Residuals — {best_name}", fontsize=13)
axes[0].set_xlabel("Predicted Salary")
axes[0].set_ylabel("Residual")

axes[1].scatter(y_test, y_pred_best, alpha=0.4, color="#7c5cd8")
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--")
axes[1].set_title("Actual vs Predicted", fontsize=13)
axes[1].set_xlabel("Actual Salary")
axes[1].set_ylabel("Predicted Salary")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/residuals_{best_name.replace(' ', '_')}.png", dpi=120)
plt.close(fig)
print(f"\n    Saved: charts/residuals_{best_name.replace(' ', '_')}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODELS & METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Saving models …")

for name, pipe in pipelines.items():
    fname = name.replace(" ", "_").lower() + "_pipeline.joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
    print(f"    Saved: models/{fname}")

# Save best model explicitly as 'best_model.joblib'
joblib.dump(best_pipe, os.path.join(MODEL_DIR, "best_model.joblib"))
print(f"    Saved: models/best_model.joblib  (best = {best_name})")

# Save metadata (categories + best model name) for app.py
metadata = {
    "best_model":      best_name,
    "edu_categories":  edu_categories,
    "job_categories":  job_categories,
    "loc_categories":  loc_categories,
    "results":         results,
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))
print("    Saved: models/metadata.joblib")


# ══════════════════════════════════════════════════════════════════════════════
# 7. QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Sanity check — sample prediction:")
sample = pd.DataFrame([{
    "age":        30,
    "education":  "Bachelor",
    "job_title":  "Data Scientist",
    "location":   "San Francisco",
    "experience": 5.0,
}])
predicted = best_pipe.predict(sample)[0]
print(f"    Input  : Age=30, Bachelor, Data Scientist, San Francisco, 5 yrs exp")
print(f"    Output : ${predicted:,.2f} / year")

print("\n" + "=" * 60)
print("  Training complete!")
print("=" * 60)
