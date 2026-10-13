"""
generate_data.py
Generates a synthetic salary dataset and saves it to data/salary_data.csv.
Run this once if salary_data.csv is missing.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 2000

education_levels = ["High School", "Associate", "Bachelor", "Master", "PhD"]
job_titles = [
    "Software Engineer", "Data Analyst", "Data Scientist", "Product Manager",
    "Marketing Specialist", "Financial Analyst", "HR Manager", "Sales Representative",
    "UX Designer", "DevOps Engineer", "Business Analyst", "Project Manager",
]
locations = ["New York", "San Francisco", "Austin", "Chicago", "Seattle",
             "Boston", "Denver", "Los Angeles", "Miami", "Dallas"]

# Base salaries per education level
edu_base = {
    "High School":  32000,
    "Associate":    40000,
    "Bachelor":     58000,
    "Master":       75000,
    "PhD":          92000,
}

# Multiplier per job title
job_mult = {
    "Software Engineer":    1.35,
    "Data Analyst":         1.15,
    "Data Scientist":       1.45,
    "Product Manager":      1.40,
    "Marketing Specialist": 0.95,
    "Financial Analyst":    1.20,
    "HR Manager":           1.05,
    "Sales Representative": 0.90,
    "UX Designer":          1.10,
    "DevOps Engineer":      1.30,
    "Business Analyst":     1.12,
    "Project Manager":      1.25,
}

# Location cost-of-living multiplier
loc_mult = {
    "New York":       1.30,
    "San Francisco":  1.45,
    "Austin":         1.05,
    "Chicago":        1.15,
    "Seattle":        1.25,
    "Boston":         1.20,
    "Denver":         1.10,
    "Los Angeles":    1.25,
    "Miami":          1.05,
    "Dallas":         1.00,
}

records = []
for _ in range(N):
    edu   = np.random.choice(education_levels, p=[0.12, 0.13, 0.40, 0.25, 0.10])
    job   = np.random.choice(job_titles)
    loc   = np.random.choice(locations)
    exp   = float(np.random.randint(0, 31))                   # years 0–30
    age   = int(max(22, min(65, exp + np.random.randint(20, 28))))  # age derived from exp

    base_salary = (
        edu_base[edu]
        * job_mult[job]
        * loc_mult[loc]
        + exp * 2200                            # each year of experience adds ~$2200
        + np.random.normal(0, 6000)             # random noise
    )
    base_salary = round(max(25000, base_salary), 2)

    records.append({
        "age":        age,
        "education":  edu,
        "job_title":  job,
        "location":   loc,
        "experience": exp,
        "salary":     base_salary,
    })

df = pd.DataFrame(records)

# Introduce ~2% missing values for realism
for col in ["experience", "age"]:
    mask = np.random.choice([True, False], size=N, p=[0.02, 0.98])
    df.loc[mask, col] = np.nan

# Introduce ~1% duplicates
dup_idx = np.random.choice(df.index, size=int(N * 0.01), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/salary_data.csv", index=False)
print(f"Dataset saved: {len(df)} rows  →  data/salary_data.csv")
