#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'data' / 'student_health_dataset_50k.csv'
OUT = ROOT / 'data' / 'student_health_dataset_50k_transformed.csv'

print(f"Input: {IN}")
print(f"Output: {OUT}\n")

if not IN.exists():
    raise FileNotFoundError(f"Input file not found: {IN}")

# Load
df = pd.read_csv(IN)
print(f"Loaded dataset shape: {df.shape}")

# Drop obvious unwanted columns if present
drop_cols = [c for c in ['student_id', 'timestamp'] if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")

# Identify target column (heuristic)
target_col = 'health_condition' if 'health_condition' in df.columns else None
if target_col:
    print(f"Detected target column: {target_col}")

# Discretize numeric columns into low/mid/high using 33/67 quantiles
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude target if numeric
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

print(f"Numeric columns to discretize: {numeric_cols}")
for col in numeric_cols:
    try:
        q1, q2 = df[col].quantile([0.33, 0.67])
        df[col] = pd.cut(df[col], bins=[-np.inf, q1, q2, np.inf], labels=['low', 'mid', 'high'], include_lowest=True)
    except Exception as e:
        print(f"Warning: could not discretize {col}: {e}")

# Optionally encode target to labels if it's categorical -- keep original values
# Drop rows with NaN produced by discretization
before = len(df)
df = df.dropna()
after = len(df)
print(f"Dropped {before-after} rows with NA after discretization. Remaining: {after}")

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Transformed dataset written to {OUT}")
