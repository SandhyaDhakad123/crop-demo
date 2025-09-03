import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('crop_data_small.csv')
df.columns = df.columns.str.strip().str.lower()

# Keep original names for dropdown
district_names = df['district'].unique()
state_names = df['state'].unique()
season_names = df['season'].unique()
soil_names = df['soil_type'].unique()

# Encode categorical features for model
le_district = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

df['district_encoded'] = le_district.fit_transform(df['district'])
df['state_encoded'] = le_state.fit_transform(df['state'])
df['season_encoded'] = le_season.fit_transform(df['season'])
df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
df['crop_encoded'] = le_crop.fit_transform(df['crop'])

# Features and target
X = df[['district_encoded','state_encoded','season_encoded','soil_encoded','potassium','temperature_c','humidity_pct','rainfall_mm']]
y = df['crop_encoded']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🌾 AI-Based Crop Recommendation System")

# Dropdowns for categorical inputs
# Dropdowns for categorical inputs (safe)
district = st.selectbox("Select District", options=df['district'].unique())
state = st.selectbox("Select State", options=df['state'].unique())
season = st.selectbox("Select Season", options=df['season'].unique())
soil_type = st.selectbox("Select Soil Type", options=df['soil_type'].unique())

# Sliders for numerical inputs
potassium = st.slider("Potassium (K) value", min_value=0, max_value=500, value=200)
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=800.0)


# Predict button
if st.button("Recommend Crop"):
    input_df = pd.DataFrame({
        'district_encoded':[le_district.transform([district])[0]],
        'state_encoded':[le_state.transform([state])[0]],
        'season_encoded':[le_season.transform([season])[0]],
        'soil_encoded':[le_soil.transform([soil_type])[0]],
        'potassium':[potassium],
        'temperature_c':[temperature],
        'humidity_pct':[humidity],
        'rainfall_mm':[rainfall]
    })
    pred = model.predict(input_df)
    pred_name = le_crop.inverse_transform(pred)
    st.success(f"🌾 Recommended Crop: {pred_name[0]}")
