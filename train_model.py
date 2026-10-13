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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
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

if not os.path.exists(DATA_PATH):
    print("\n[INFO] Dataset not found — running generate_data.py …")
    import generate_data  # noqa: F401

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Raw dataset shape : {df.shape}")
print(df.head())

before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n    Duplicates removed  : {before - len(df)}")

for col in ["experience", "age"]:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    df[col] = df[col].astype(float)

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

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df["salary"], bins=40, kde=True, color="#3b82d4", ax=ax)
ax.set_title("Annual Salary Distribution", fontsize=14)
ax.set_xlabel("Annual Salary (USD)")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_distribution.png", dpi=120)
plt.close(fig)

edu_order = ["High School", "Associate", "Bachelor", "Master", "PhD"]
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=df, x="education", y="salary", order=edu_order, palette="Blues", ax=ax)
ax.set_title("Salary by Education Level", fontsize=14)
ax.set_xlabel("Education Level")
ax.set_ylabel("Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_by_education.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
sns.scatterplot(data=df, x="experience", y="salary", alpha=0.4, color="#7c5cd8", ax=ax)
sns.regplot(data=df, x="experience", y="salary", scatter=False, color="red", ax=ax)
ax.set_title("Salary vs Years of Experience", fontsize=14)
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_vs_experience.png", dpi=120)
plt.close(fig)

med_job = df.groupby("job_title")["salary"].median().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
med_job.plot(kind="barh", color="#3b82d4", ax=ax)
ax.set_title("Median Salary by Job Title", fontsize=14)
ax.set_xlabel("Median Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_by_job.png", dpi=120)
plt.close(fig)

med_loc = df.groupby("location")["salary"].median().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
med_loc.plot(kind="barh", color="#7c5cd8", ax=ax)
ax.set_title("Median Salary by Location", fontsize=14)
ax.set_xlabel("Median Annual Salary (USD)")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/salary_by_location.png", dpi=120)
plt.close(fig)

corr_exp = df["experience"].corr(df["salary"])
print(f"      • Median salary : ${df['salary'].median():>10,.0f}")
print(f"      • Experience↔Salary corr: {corr_exp:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building preprocessing pipeline …")

FEATURES = ["age", "education", "job_title", "location", "experience"]
TARGET   = "salary"

X = df[FEATURES]
y = df[TARGET]

CAT_COLS = ["education", "job_title", "location"]
NUM_COLS = ["age", "experience"]

edu_categories = ["High School", "Associate", "Bachelor", "Master", "PhD"]
job_categories = sorted(df["job_title"].unique().tolist())
loc_categories = sorted(df["location"].unique().tolist())

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OrdinalEncoder(
        categories=[edu_categories, job_categories, loc_categories],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ), CAT_COLS),
], remainder="drop")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess outside any pipeline to avoid sklearn 1.6 / Python 3.14 validate_data crash
preprocessor.fit(X_train)
X_train_t = np.ascontiguousarray(preprocessor.transform(X_train), dtype=np.float64)
X_test_t  = np.ascontiguousarray(preprocessor.transform(X_test),  dtype=np.float64)
y_train_a = np.ascontiguousarray(y_train.values, dtype=np.float64)
y_test_a  = np.ascontiguousarray(y_test.values,  dtype=np.float64)
print(f"    Train size: {len(X_train_t)}  |  Test size: {len(X_test_t)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Training models …")

# Linear Regression via numpy lstsq — avoids all sklearn linear model bugs on Python 3.14
X_bias = np.column_stack([np.ones(len(X_train_t)), X_train_t])
coeffs, _, _, _ = np.linalg.lstsq(X_bias, y_train_a, rcond=None)

def lr_predict(X_arr):
    return np.column_stack([np.ones(len(X_arr)), X_arr]) @ coeffs

lr_pred_test = lr_predict(X_test_t)
lr_mae  = mean_absolute_error(y_test_a, lr_pred_test)
lr_rmse = np.sqrt(mean_squared_error(y_test_a, lr_pred_test))
lr_r2   = r2_score(y_test_a, lr_pred_test)

print(f"\n    Linear Regression (numpy OLS)")
print(f"      MAE  : ${lr_mae:>10,.2f}")
print(f"      RMSE : ${lr_rmse:>10,.2f}")
print(f"      R²   : {lr_r2:.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_t, y_train_a)
rf_pred_test = rf.predict(X_test_t)
rf_mae  = mean_absolute_error(y_test_a, rf_pred_test)
rf_rmse = np.sqrt(mean_squared_error(y_test_a, rf_pred_test))
rf_r2   = r2_score(y_test_a, rf_pred_test)

print(f"\n    Random Forest")
print(f"      MAE  : ${rf_mae:>10,.2f}")
print(f"      RMSE : ${rf_rmse:>10,.2f}")
print(f"      R²   : {rf_r2:.4f}")

results = {
    "Linear Regression": {"MAE": lr_mae, "RMSE": lr_rmse, "R2": lr_r2},
    "Random Forest":     {"MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2},
}

# Wrapper classes so saved objects accept raw DataFrames at prediction time
class LinearRegressionWrapper:
    def __init__(self, preprocessor, coeffs):
        self.preprocessor = preprocessor
        self.coeffs = coeffs
    def predict(self, X):
        Xt = np.ascontiguousarray(self.preprocessor.transform(X), dtype=np.float64)
        return np.column_stack([np.ones(len(Xt)), Xt]) @ self.coeffs

class RandomForestWrapper:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    def predict(self, X):
        Xt = np.ascontiguousarray(self.preprocessor.transform(X), dtype=np.float64)
        return self.model.predict(Xt)

pipelines = {
    "Linear Regression": LinearRegressionWrapper(preprocessor, coeffs),
    "Random Forest":     RandomForestWrapper(preprocessor, rf),
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESIDUAL PLOT
# ══════════════════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]["R2"])
best_pipe = pipelines[best_name]
y_pred_best = best_pipe.predict(X_test)
residuals   = y_test_a - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.4, color="#3b82d4")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title(f"Residuals — {best_name}", fontsize=13)
axes[0].set_xlabel("Predicted Salary")
axes[0].set_ylabel("Residual")
axes[1].scatter(y_test_a, y_pred_best, alpha=0.4, color="#7c5cd8")
axes[1].plot([y_test_a.min(), y_test_a.max()],
             [y_test_a.min(), y_test_a.max()], "r--")
axes[1].set_title("Actual vs Predicted", fontsize=13)
axes[1].set_xlabel("Actual Salary")
axes[1].set_ylabel("Predicted Salary")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/residuals_{best_name.replace(' ', '_')}.png", dpi=120)
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODELS & METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Saving models …")

for name, pipe in pipelines.items():
    fname = name.replace(" ", "_").lower() + "_pipeline.joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
    print(f"    Saved: models/{fname}")

joblib.dump(best_pipe, os.path.join(MODEL_DIR, "best_model.joblib"))
print(f"    Saved: models/best_model.joblib  (best = {best_name})")

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
# 7. SANITY CHECK
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
