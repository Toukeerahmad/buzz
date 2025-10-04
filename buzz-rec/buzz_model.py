# buzz_model.py
"""
Train a BuzzLocator model (Linear Regression & RandomForest comparison),
create a heuristic SurvivalScore if missing, and save the best model and preprocessor.

Streamlit Cloud compatible:
- Saves with joblib protocol=4
- Avoids pickling internal sklearn classes that may break on cloud
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "buzz-complete.csv"
MODEL_PATH = BASE_DIR / "buzz_model.pkl"
REPORT_PATH = BASE_DIR / "training_report.txt"

assert DATA_PATH.exists(), f"CSV not found at {DATA_PATH}"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)

# Detect columns
def detect_cols(df):
    cols = {c.lower(): c for c in df.columns}
    def find(candidates):
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    business_col = find(['business','category','type','business_type','DemandProducts'])
    area_col = find(['area','neighborhood','location','city','locality','district','taluk','Area'])
    
    candidates_pos = ['demand','footfall','population','income','sales','rating','popularity']
    candidates_neg = ['competition','competitors','rent','crime','cost','cost_of_living','avg_rent']
    pos_cols = [cols[c] for c in cols if any(p in c for p in candidates_pos)]
    neg_cols = [cols[c] for c in cols if any(n in c for n in candidates_neg)]
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    return {
        'business_col': business_col,
        'area_col': area_col,
        'pos_cols': pos_cols,
        'neg_cols': neg_cols,
        'numeric_cols': numeric_cols
    }

det = detect_cols(df)
business_col = det['business_col']
area_col = det['area_col']
pos_cols = det['pos_cols']
neg_cols = det['neg_cols']
numeric_cols = det['numeric_cols']

# -------------------------
# SurvivalScore generator
# -------------------------
def make_survival_score(df, business_col=None, area_col=None, pos_cols=None, neg_cols=None):
    pos_cols = pos_cols or []
    neg_cols = neg_cols or []

    F = pd.DataFrame(index=df.index)
    for c in (pos_cols + neg_cols):
        if c in df.columns:
            col_numeric = pd.to_numeric(df[c], errors='coerce')
            F[c] = col_numeric.fillna(col_numeric.median()) if np.issubdtype(col_numeric.dtype, np.number) else 0
    if F.shape[1] == 0:
        F['dummy'] = 1.0

    F_norm = (F - F.min()) / (F.max() - F.min() + 1e-9)
    pos_sum = F_norm[[c for c in F_norm.columns if c in pos_cols]].sum(axis=1)
    neg_sum = F_norm[[c for c in F_norm.columns if c in neg_cols]].sum(axis=1)
    base_score = pos_sum - neg_sum

    if business_col and business_col in df.columns:
        business_mean = base_score.groupby(df[business_col]).transform('mean')
        base_score = 0.7 * base_score + 0.3 * business_mean
    if area_col and area_col in df.columns:
        area_mean = base_score.groupby(df[area_col]).transform('mean')
        base_score = 0.8 * base_score + 0.2 * area_mean

    minv, maxv = base_score.min(), base_score.max()
    score = 100 * (base_score - minv) / (maxv - minv + 1e-9)
    return score.round(2)

# Check if dataset has existing target
existing_target = None
for candidate in ['SurvivalScore', 'survival_score', 'Success', 'success', 'rating', 'Score', 'score']:
    if candidate in df.columns:
        existing_target = candidate
        break

if existing_target:
    y = pd.to_numeric(df[existing_target], errors='coerce').fillna(df[existing_target].median())
else:
    print("No existing target found — creating synthetic SurvivalScore.")
    y = make_survival_score(df, business_col, area_col, pos_cols, neg_cols)
    df['SurvivalScore'] = y

# -------------------------
# Feature selection
# -------------------------
use_numeric = [c for c in numeric_cols if c != existing_target]
if len(use_numeric) == 0:
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            use_numeric.append(c)
        except Exception:
            pass

categorical_cols = [col for col in [area_col, business_col] if col]
FEATURE_COLS = use_numeric + categorical_cols
FEATURE_COLS = [c for c in FEATURE_COLS if c != 'SurvivalScore']
X = df[FEATURE_COLS].copy()

if X.shape[1] == 0:
    raise RuntimeError("No features available to train on.")

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

for name, model_instance in models.items():
    model_instance.fit(X_train, y_train)
    preds = model_instance.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    report_lines.append(f"{name}: R2={r2:.4f}, MAE={mae:.4f}")
    print(report_lines[-1])
    if r2 > best_score:
        best_score = r2
        best = (name, model_instance)

# -------------------------
# Save model bundle (Streamlit Cloud compatible)
# -------------------------
save_bundle = {
    'model_name': best[0],
    'model': best[1],
    'preprocessor': preprocessor,
    'feature_cols': FEATURE_COLS,
    'area_col': area_col,
    'business_col': business_col
}

joblib.dump(save_bundle, MODEL_PATH, protocol=4)  # protocol=4 fixes Streamlit Cloud pickle issue

report_lines.append(f"Best model: {best[0]} (saved to {MODEL_PATH})")
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print("\nTraining complete. Artifacts written to:")
print(f" - model bundle: {MODEL_PATH}")
print(f" - report:       {REPORT_PATH}")
