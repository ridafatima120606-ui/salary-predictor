"""
train_model.py — Salary Predictor Training Script
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_wrapper import RandomForestWrapper

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

DATA_PATH = "data/salary_data.csv"
MODEL_DIR = "models"
CHART_DIR = "charts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

print("=" * 60)
print("  Salary Predictor — Training Script")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    import generate_data

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Raw shape: {df.shape}")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"    Duplicates removed: {before - len(df)}")

for col in ["experience", "age"]:
    df[col].fillna(df[col].median(), inplace=True)
    df[col] = df[col].astype(float)

df["salary"]     = pd.to_numeric(df["salary"],     errors="coerce")
df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
df["age"]        = pd.to_numeric(df["age"],        errors="coerce")
df.dropna(subset=["salary"], inplace=True)
print(f"    Clean shape: {df.shape}")
print(df["salary"].describe().to_string())

print("\n[2] Generating EDA charts ...")

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df["salary"], bins=40, kde=True, color="#3b82d4", ax=ax)
ax.set_title("Annual Salary Distribution", fontsize=14)
ax.set_xlabel("Annual Salary (USD)"); ax.set_ylabel("Count")
plt.tight_layout(); fig.savefig(f"{CHART_DIR}/salary_distribution.png", dpi=120); plt.close(fig)

edu_order = ["High School", "Associate", "Bachelor", "Master", "PhD"]
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=df, x="education", y="salary", order=edu_order, palette="Blues", ax=ax)
ax.set_title("Salary by Education Level", fontsize=14)
plt.tight_layout(); fig.savefig(f"{CHART_DIR}/salary_by_education.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
sns.scatterplot(data=df, x="experience", y="salary", alpha=0.4, color="#7c5cd8", ax=ax)
sns.regplot(data=df, x="experience", y="salary", scatter=False, color="red", ax=ax)
ax.set_title("Salary vs Years of Experience", fontsize=14)
plt.tight_layout(); fig.savefig(f"{CHART_DIR}/salary_vs_experience.png", dpi=120); plt.close(fig)

med_job = df.groupby("job_title")["salary"].median().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
med_job.plot(kind="barh", color="#3b82d4", ax=ax)
ax.set_title("Median Salary by Job Title", fontsize=14)
plt.tight_layout(); fig.savefig(f"{CHART_DIR}/salary_by_job.png", dpi=120); plt.close(fig)

med_loc = df.groupby("location")["salary"].median().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
med_loc.plot(kind="barh", color="#7c5cd8", ax=ax)
ax.set_title("Median Salary by Location", fontsize=14)
plt.tight_layout(); fig.savefig(f"{CHART_DIR}/salary_by_location.png", dpi=120); plt.close(fig)

print(f"    Median salary: ${df['salary'].median():,.0f}")
print(f"    Experience-Salary corr: {df['experience'].corr(df['salary']):.3f}")

print("\n[3] Preprocessing ...")

FEATURES = ["age", "education", "job_title", "location", "experience"]
TARGET   = "salary"
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

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor.fit(X_train)
X_train_t = np.ascontiguousarray(preprocessor.transform(X_train), dtype=np.float64)
X_test_t  = np.ascontiguousarray(preprocessor.transform(X_test),  dtype=np.float64)
y_train_a = np.ascontiguousarray(y_train.values, dtype=np.float64)
y_test_a  = np.ascontiguousarray(y_test.values,  dtype=np.float64)
print(f"    Train: {len(X_train_t)}  Test: {len(X_test_t)}")

print("\n[4] Training Random Forest ...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_t, y_train_a)
rf_pred = rf.predict(X_test_t)
rf_mae  = mean_absolute_error(y_test_a, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test_a, rf_pred))
rf_r2   = r2_score(y_test_a, rf_pred)

print(f"    MAE  : ${rf_mae:,.2f}")
print(f"    RMSE : ${rf_rmse:,.2f}")
print(f"    R2   : {rf_r2:.4f}")

results = {"Random Forest": {"MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2}}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
residuals = y_test_a - rf_pred
axes[0].scatter(rf_pred, residuals, alpha=0.4, color="#3b82d4")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals - Random Forest", fontsize=13)
axes[0].set_xlabel("Predicted Salary"); axes[0].set_ylabel("Residual")
axes[1].scatter(y_test_a, rf_pred, alpha=0.4, color="#7c5cd8")
axes[1].plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], "r--")
axes[1].set_title("Actual vs Predicted", fontsize=13)
axes[1].set_xlabel("Actual Salary"); axes[1].set_ylabel("Predicted Salary")
plt.tight_layout()
fig.savefig(f"{CHART_DIR}/residuals_Random_Forest.png", dpi=120)
plt.close(fig)

print("\n[5] Saving ...")

best_pipe = RandomForestWrapper(preprocessor, rf)
joblib.dump(best_pipe, os.path.join(MODEL_DIR, "best_model.joblib"))
print("    Saved: models/best_model.joblib")

metadata = {
    "best_model":     "Random Forest",
    "edu_categories": edu_categories,
    "job_categories": job_categories,
    "loc_categories": loc_categories,
    "results":        results,
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))
print("    Saved: models/metadata.joblib")

print("\n[6] Sanity check ...")
sample = pd.DataFrame([{
    "age": 30, "education": "Bachelor",
    "job_title": "Data Scientist",
    "location": "San Francisco", "experience": 5.0,
}])
print(f"    Predicted: ${best_pipe.predict(sample)[0]:,.2f} / year")

print("\n" + "=" * 60)
print("  Training complete!")
print("=" * 60)
