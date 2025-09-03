import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
df = pd.read_csv('crop_data_small.csv')

# 2️⃣ Clean column names
df.columns = df.columns.str.strip().str.lower()  # remove spaces & lowercase

# 3️⃣ Encode categorical features
le = LabelEncoder()
df['district'] = le.fit_transform(df['district'])
df['state'] = le.fit_transform(df['state'])
df['season'] = le.fit_transform(df['season'])
df['soil_type'] = le.fit_transform(df['soil_type'])

# 4️⃣ Encode target column (crop)
crop_le = LabelEncoder()
df['crop_encoded'] = crop_le.fit_transform(df['crop'])

# 5️⃣ Define Features & Target
X = df[['district','state','season','soil_type','potassium','temperature_c','humidity_pct','rainfall_mm']]
y = df['crop_encoded']

# 6️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Test & Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# 9️⃣ Example prediction
example = pd.DataFrame({
    'district':[df['district'][0]],       # use already encoded value
    'state':[df['state'][0]],
    'season':[df['season'][0]],
    'soil_type':[df['soil_type'][0]],
    'potassium':[200],
    'temperature_c':[28.0],
    'humidity_pct':[65.0],
    'rainfall_mm':[800.0]
})

pred_crop = model.predict(example)
pred_crop_name = crop_le.inverse_transform(pred_crop)
print(f"🌾 Recommended Crop: {pred_crop_name[0]}")
