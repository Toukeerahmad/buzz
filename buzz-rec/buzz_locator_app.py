# buzz_locator_app.py
"""
Streamlit app to query the BuzzLocator model.
Assumes you have trained and saved the model bundle at /mnt/data/buzz_model.pkl
Run with:
  streamlit run buzz_locator_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_BUNDLE = Path("buzz_model.pkl")
CSV_PATH = Path("buzz-complete.csv")

st.set_page_config(page_title="BuzzLocator", layout="wide")
st.title("BuzzLocator — Business Location Recommender")

if not MODEL_BUNDLE.exists():
    st.error(f"Model bundle not found at {MODEL_BUNDLE}. Run buzz_model.py first.")
    st.stop()

bundle = joblib.load(MODEL_BUNDLE)
model_name = bundle['model_name']
model = bundle['model']
preprocessor = bundle['preprocessor']
FEATURE_COLS = bundle['feature_cols']
area_col = bundle.get('area_col', None)
business_col = bundle.get('business_col', None)

st.write(f"Using model: **{model_name}**")

# Load dataset for area-level aggregation & display
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=FEATURE_COLS)  # empty fallback

# UI inputs
st.sidebar.header("User inputs")
preferred_business = st.sidebar.text_input("Preferred business (optional)", "")
preferred_area = st.sidebar.text_input("Preferred area (optional)", "")
multiselect_areas = []
if area_col and area_col in df.columns:
    all_areas = sorted(df[area_col].dropna().astype(str).unique().tolist())
    multiselect_areas = st.sidebar.multiselect("Compare areas (pick multiple)", options=all_areas, default=all_areas[:5])
else:
    st.sidebar.info("No 'area' column detected in dataset; app will operate per-row.")

top_k = st.sidebar.slider("How many top areas to show (overall)", 1, 20, 5)

# Prepare candidate rows to predict:
# If user gave a preferred area, we compute aggregated average features for that area.
def make_feature_rows_for_areas(df, areas):
    rows = []
    for a in areas:
        subset = df[df[area_col].astype(str) == str(a)] if (area_col and area_col in df.columns) else pd.DataFrame()
        if subset.shape[0] == 0:
            # if no rows, create a row with NaNs so preprocessor imputes with medians
            row = {c:np.nan for c in FEATURE_COLS}
            if area_col:
                row[area_col] = a
            if business_col:
                row[business_col] = preferred_business or np.nan
        else:
            # average numeric columns, and for categorical use the most common
            row = {}
            for c in FEATURE_COLS:
                if c in subset.columns:
                    if pd.api.types.is_numeric_dtype(subset[c]):
                        row[c] = subset[c].mean()
                    else:
                        # most frequent
                        row[c] = subset[c].mode().iloc[0] if not subset[c].mode().empty else subset[c].iloc[0]
                else:
                    row[c] = np.nan
            if business_col:
                row[business_col] = preferred_business or row.get(business_col, np.nan)
        # optionally tag the preferred business
        if business_col and preferred_business:
            row[business_col] = preferred_business
        rows.append(row)
    return pd.DataFrame(rows)

# Build prediction dataframe
prediction_df = None
if multiselect_areas:
    prediction_df = make_feature_rows_for_areas(df, multiselect_areas)
    labels = multiselect_areas
elif preferred_area and area_col and area_col in df.columns:
    prediction_df = make_feature_rows_for_areas(df, [preferred_area])
    labels = [preferred_area]
else:
    # fallback: show top_k aggregated by area from dataset (if area exists) else top rows
    if area_col and area_col in df.columns:
        area_scores_df = df.groupby(area_col).apply(lambda g: g.mean(numeric_only=True)).reset_index()
        top_areas = area_scores_df.sort_values(by=area_scores_df.columns[1], ascending=False)[area_col].astype(str).tolist()[:top_k]
        prediction_df = make_feature_rows_for_areas(df, top_areas)
        labels = top_areas
    else:
        # No area info: operate per-row. Use top_k rows from dataset.
        sample = df.head(top_k)
        if sample.shape[0] == 0:
            st.warning("Dataset appears empty or has no usable features.")
            prediction_df = pd.DataFrame([{c:np.nan for c in FEATURE_COLS}])
            labels = ["Unknown"]
        else:
            rows = []
            lbls = []
            for idx, row in sample.iterrows():
                r = {}
                for c in FEATURE_COLS:
                    r[c] = row[c] if c in row else np.nan
                if business_col and preferred_business:
                    r[business_col] = preferred_business
                rows.append(r)
                lbls.append(str(row[area_col]) if area_col in row else f"row-{idx}")
            prediction_df = pd.DataFrame(rows)
            labels = lbls

# Apply preprocessor and predict
X_prepared = preprocessor.transform(prediction_df[FEATURE_COLS])
preds = model.predict(X_prepared)
preds = np.clip(preds, 0, 100)  # ensure 0-100 range

result_df = pd.DataFrame({
    'area': labels,
    'predicted_survival_score': np.round(preds,2)
})
result_df = result_df.sort_values('predicted_survival_score', ascending=False)

# Show results
st.subheader("Predicted survival scores")
st.write("Higher score (0-100) -> better chance of business surviving there (heuristic & learned).")
st.dataframe(result_df.reset_index(drop=True))

# Bar chart comparison
st.subheader("Comparison chart")
fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(result_df))))
ax.barh(result_df['area'].astype(str)[::-1], result_df['predicted_survival_score'][::-1])
ax.set_xlabel("Predicted survival score (0-100)")
ax.set_title("Area comparison — higher is better")
plt.tight_layout()
st.pyplot(fig)

# Pros / Cons logic per area (simple heuristic)
st.subheader("Pros & Cons (automatically inferred from dataset medians)")
proscons = []
# compute medians once
medians = {}
for c in FEATURE_COLS:
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
        medians[c] = df[c].median()

for i, row in result_df.iterrows():
    area_label = row['area']
    # fetch aggregated row we computed earlier
    idx = list(labels).index(area_label) if area_label in labels else None
    if idx is not None:
        feat_row = prediction_df.iloc[idx]
    else:
        feat_row = pd.Series({c:np.nan for c in FEATURE_COLS})

    pros = []
    cons = []
    # Look for positive indicators: any numeric feature in FEATURE_COLS that is above median and named like demand/footfall/income
    for c in FEATURE_COLS:
        lc = c.lower()
        if c not in feat_row.index:
            continue
        try:
            val = float(feat_row[c])
        except Exception:
            continue
        med = medians.get(c, None)
        if med is None:
            continue
        if any(k in lc for k in ['demand','foot','popul','income','sale','rating']):
            if val >= med:
                pros.append(f"{c} above median")
        if any(k in lc for k in ['comp','compet','rent','crime','cost']):
            if val >= med:
                cons.append(f"{c} high (>= median)")
    if preferred_business and business_col:
        pros.append("Matches preferred business")
    if not pros: pros = ["No strong pros detected from available columns"]
    if not cons: cons = ["No strong cons detected from available columns"]
    proscons.append({'area': area_label, 'pros': "; ".join(pros), 'cons': "; ".join(cons)})

pc_df = pd.DataFrame(proscons).sort_values('area')
st.dataframe(pc_df)

# Top suggestions
st.subheader("Top suggestions")
st.write(f"Top {min(len(result_df), top_k)} areas by predicted survival score:")
st.table(result_df.head(top_k).reset_index(drop=True))

st.markdown("""
**Method notes:**  
- The underlying model was trained on your dataset (`buzz-complete.csv`) using both a Linear Regression and a Random Forest.  
- The app aggregates area-level features (mean of numeric features, mode of categorical) and predicts a survival score (0–100).  
- This is a data-driven recommendation — quality depends on features present in your CSV (population, demand-like columns, footfall, rent, crime, etc.).  
""")
