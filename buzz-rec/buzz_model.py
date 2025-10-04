# buzz_model.py
"""
Train a BuzzLocator model (Linear Regression & RandomForest comparison),
create a heuristic SurvivalScore if missing, and save the best model and preprocessor.
Outputs:
  - /mnt/data/buzz_model.pkl           (best trained model)
  - /mnt/data/buzz_preprocessor.pkl    (fitted preprocessing pipeline & feature list)
  - /mnt/data/training_report.txt      (brief training metrics)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = Path("buzz-complete.csv")
MODEL_PATH = Path("buzz_model.pkl")
PREPROC_PATH = Path("buzz_preprocessor.pkl")
REPORT_PATH = Path("training_report.txt")

assert DATA_PATH.exists(), f"CSV not found at {DATA_PATH}"

# -------------------------
# Utility: detect columns
# -------------------------
def detect_cols(df):
    cols = {c.lower(): c for c in df.columns}
    # heuristics
    def find(candidates):
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    business_col = find(['business','category','type','business_type','DemandProducts'])
    area_col = find(['area','neighborhood','location','city','locality','district','taluk','Area'])
    # numeric semantic candidates
    candidates_pos = ['demand','footfall','population','income','sales','rating','popularity']
    candidates_neg = ['competition','competitors','rent','crime','cost','cost_of_living','cost_o_living','avg_rent']
    pos_cols = [cols[c] for c in cols if any(p in c for p in candidates_pos)]
    neg_cols = [cols[c] for c in cols if any(n in c for n in candidates_neg)]
    # numeric columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    # prefer semantic columns but fallback to numeric set
    return {
        'business_col': business_col,
        'area_col': area_col,
        'pos_cols': pos_cols,
        'neg_cols': neg_cols,
        'numeric_cols': numeric_cols
    }

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)
df_orig = df.copy()
det = detect_cols(df)
business_col = det['business_col']
area_col = det['area_col']
pos_cols = det['pos_cols']
neg_cols = det['neg_cols']
numeric_cols = det['numeric_cols']

# -------------------------
# Build synthetic SurvivalScore
# -------------------------
# def make_survival_score(df):
#     # choose features to include in the synthetic target
#     # positive features: those with names suggesting demand/footfall/income
#     pos = pos_cols.copy()
#     neg = neg_cols.copy()

#     # If none detected, fall back to heuristics using numeric columns:
#     if len(pos) == 0 and len(neg) == 0:
#         # use half of numeric columns as positive, rest as negative heuristically
#         if len(numeric_cols) == 0:
#             # no numeric data: return neutral 50
#             return pd.Series(50.0, index=df.index)
#         pos = numeric_cols[: max(1, len(numeric_cols)//2)]
#         neg = numeric_cols[max(1,len(numeric_cols)//2):]

#     # Convert to numeric and fillna
#     F = pd.DataFrame(index=df.index)
#     for c in pos + neg:
#         if c in df.columns:
#             F[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median() if df[c].dtype != object else 0)
#     if F.shape[1] == 0:
#         return pd.Series(50.0, index=df.index)

#     # Normalize each column (0..1)
#     F_norm = (F - F.min()) / (F.max() - F.min() + 1e-9)

#     # Build weighted sum: pos features +1, neg features -1
#     pos_sum = F_norm[[c for c in F_norm.columns if c in pos]].sum(axis=1) if len(pos) > 0 else 0
#     neg_sum = F_norm[[c for c in F_norm.columns if c in neg]].sum(axis=1) if len(neg) > 0 else 0

#     raw_score = pos_sum - neg_sum

#     # If business column present, give a tiny boost to rows that already match common businesses (not required)
#     # scale to 0-100
#     minv, maxv = raw_score.min(), raw_score.max()
#     if maxv - minv < 1e-9:
#         score = pd.Series(50.0, index=df.index)
#     else:
#         score = 100 * (raw_score - minv) / (maxv - minv)
#     return score.round(2)

# # If the user dataset already had a SurvivalScore-like column, prefer that
# existing_target = None
# for candidate in ['SurvivalScore','survival_score','Success','success','rating','Score','score']:
#     if candidate in df.columns:
#         existing_target = candidate
#         break

# if existing_target:
#     y = pd.to_numeric(df[existing_target], errors='coerce').fillna(df[existing_target].median())
# else:
#     print("No existing target found — creating synthetic SurvivalScore.")
#     y = make_survival_score(df)
#     df['SurvivalScore'] = y





# -------------------------
# Improved SurvivalScore Generator
# -------------------------
# -------------------------
# Improved SurvivalScore Generator (Fixed for non-numeric columns)
# -------------------------
def make_survival_score(df, business_col=None, area_col=None, pos_cols=None, neg_cols=None):
    pos_cols = pos_cols or []
    neg_cols = neg_cols or []

    # create numeric features
    F = pd.DataFrame(index=df.index)
    for c in (pos_cols + neg_cols):
        if c in df.columns:
            # Try converting to numeric; if fails, fill with 0
            col_numeric = pd.to_numeric(df[c], errors='coerce')
            if np.issubdtype(col_numeric.dtype, np.number):
                F[c] = col_numeric.fillna(col_numeric.median())
            else:
                F[c] = 0  # non-numeric columns contribute 0

    if F.shape[1] == 0:
        F['dummy'] = 1.0

    # Normalize numeric features
    F_norm = (F - F.min()) / (F.max() - F.min() + 1e-9)
    pos_sum = F_norm[[c for c in F_norm.columns if c in pos_cols]].sum(axis=1)
    neg_sum = F_norm[[c for c in F_norm.columns if c in neg_cols]].sum(axis=1)
    base_score = pos_sum - neg_sum

    # Add business-type bias
    if business_col and business_col in df.columns:
        business_mean = base_score.groupby(df[business_col]).transform('mean')
        base_score = 0.7 * base_score + 0.3 * business_mean

    # Add area bias
    if area_col and area_col in df.columns:
        area_mean = base_score.groupby(df[area_col]).transform('mean')
        base_score = 0.8 * base_score + 0.2 * area_mean

    # Scale to 0–100
    minv, maxv = base_score.min(), base_score.max()
    score = 100 * (base_score - minv) / (maxv - minv + 1e-9)
    return score.round(2)
  


# If the dataset already has a score, use it; otherwise, synthesize with the improved function
existing_target = None
for candidate in ['SurvivalScore', 'survival_score', 'Success', 'success', 'rating', 'Score', 'score']:
    if candidate in df.columns:
        existing_target = candidate
        break

if existing_target:
    y = pd.to_numeric(df[existing_target], errors='coerce').fillna(df[existing_target].median())
else:
    print("No existing target found — creating improved synthetic SurvivalScore.")
    y = make_survival_score(df, business_col, area_col, pos_cols, neg_cols)
    df['SurvivalScore'] = y

# -------------------------
# Prepare feature set X
# -------------------------
# We'll use:
#  - All numeric columns
#  - plus a few categorical columns (area, business) if present (encoded)
use_numeric = [c for c in numeric_cols if c != existing_target]
# ensure at least one numeric
if len(use_numeric) == 0:
    # attempt to coerce some columns
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            use_numeric.append(c)
        except Exception:
            pass

categorical_cols = []
if area_col:
    categorical_cols.append(area_col)
if business_col:
    categorical_cols.append(business_col)

# final selected features
FEATURE_COLS = use_numeric + categorical_cols
# drop columns that equal target
FEATURE_COLS = [c for c in FEATURE_COLS if c != 'SurvivalScore']

X = df[FEATURE_COLS].copy()
# Small check
if X.shape[1] == 0:
    raise RuntimeError("No features available to train on. Inspect CSV and column detection.")

# -------------------------
# Preprocessing pipeline
# -------------------------
num_imputer = SimpleImputer(strategy='median')
num_scaler = StandardScaler()
cat_imputer = SimpleImputer(strategy='constant', fill_value='__MISSING__')
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


num_cols = [c for c in FEATURE_COLS if c in use_numeric]
cat_cols = [c for c in FEATURE_COLS if c in categorical_cols]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([('impute', num_imputer), ('scale', num_scaler)]), num_cols),
    ("cat", Pipeline([('impute', cat_imputer), ('ohe', cat_encoder)]), cat_cols)
], remainder='drop')

# Fit preprocessor and transform
X_p = preprocessor.fit_transform(X)

# -------------------------
# Train/test split & models
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_p, y.values, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}

best = None
best_score = -np.inf
report_lines = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    report_lines.append(f"{name}: R2={r2:.4f}, MAE={mae:.4f}")
    print(report_lines[-1])
    if r2 > best_score:
        best_score = r2
        best = (name, model)

# -------------------------
# Save best model + preprocessor + feature metadata
# -------------------------
save_bundle = {
    'model_name': best[0],
    'model': best[1],
    'preprocessor': preprocessor,
    'feature_cols': FEATURE_COLS,
    'area_col': area_col,
    'business_col': business_col
}
joblib.dump(save_bundle, MODEL_PATH)
joblib.dump(preprocessor, PREPROC_PATH)

report_lines.append(f"Best model: {best[0]} (saved to {MODEL_PATH})")
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print("\nTraining complete. Artifacts written to:")
print(" - model bundle:", MODEL_PATH)
print(" - preprocessor: ", PREPROC_PATH)
print(" - report:       ", REPORT_PATH)
