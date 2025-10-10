# BuzzLocator — AI Business Location Recommender

BuzzLocator is an AI-powered system that recommends the best business locations for entrepreneurs. It ingests a dataset of areas and local indicators and outputs the most promising locations, visualized as a table, bar chart, and interactive map.

## Dataset
Place `buzz-complete.csv` in the `buzz-rec/` folder (same folder as the scripts). Expected columns include:

- District, Taluk, Area, DemandProducts
- Cost_of_Living, Population, Avg_Income
- Competition_Index, Footfall_Potential, AvgRent2BHK, MetroAccess
- CrimeReportedCitywide, CrimeDetectedCitywide

Minimal variations are handled (e.g., `CrimeReportedCitywide` values like `3603 (28.5%)`).

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Train the model
From the `buzz-rec/` directory, run:
```bash
python buzz_model.py
```
This will:
- Engineer features (`metro_binary`, `demand_token_count`)
- Compute a normalized `success_score` using the weighted formula
- Train a RandomForestRegressor
- Save the model to `buzz_model.pkl`
- Save processed data to `buzz-processed.csv`

## Run the app
From the `buzz-rec/` directory, start Streamlit:
```bash
streamlit run streamlit_app.py
```

You’ll see:
- Title and description
- Business type selector and custom input
- Slider to choose Top N (default 5)
- Table of top recommended locations
- Bar chart comparing predicted success ratios
- Folium-powered interactive map; markers use approximate coordinates if real coordinates are unavailable

## Features
- ✅ Predicts best business locations based on local conditions
- ✅ Uses RandomForest for success scoring
- ✅ Interactive bar chart visualization
- ✅ Folium-based map to visualize top N business spots
- ✅ Fully offline, easy to retrain with updated data

## Notes
- Heavy operations are cached with `@st.cache_data` and `@st.cache_resource`.
- If the model is missing, run `python buzz_model.py` before launching the app.