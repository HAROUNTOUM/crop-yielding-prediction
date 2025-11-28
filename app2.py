import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

print("🚀 Starting Fast Training...")

# 1. Load Data
file_path = 'crop_yield_augmented.csv'
if not os.path.exists(file_path):
    print(f"❌ Error: '{file_path}' not found. Using 'crop_yield.csv' instead.")
    file_path = 'crop_yield.csv'

df = pd.read_csv(file_path)

# 2. Data Cleaning
for col in ['Crop', 'State', 'Season']:
    df[col] = df[col].str.strip().str.title()

df['Season'] = df['Season'].str.strip()

# Calculate Yield
if 'Production' in df.columns and 'Area' in df.columns:
    df['Yield'] = df['Production'] / df['Area']
df = df.dropna(subset=['Yield'])

# 3. Save Helper Files (Medians & Unique lists)
print("💾 Saving metadata...")
median_values = {
    'Crop_Year': df['Crop_Year'].median(),
    'Annual_Rainfall': df['Annual_Rainfall'].median(),
    'Fertilizer': df['Fertilizer'].median() if 'Fertilizer' in df.columns else 0,
    'Pesticide': df['Pesticide'].median() if 'Pesticide' in df.columns else 0,
}
joblib.dump(median_values, 'median_values.pkl')

unique_values = {
    'crops': sorted(df['Crop'].unique().tolist()),
    'states': sorted(df['State'].unique().tolist()),
    'seasons': sorted(df['Season'].unique().tolist()),
}
joblib.dump(unique_values, 'unique_values.pkl')

# 4. Label Encoding
le_crop = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()

df['Crop'] = le_crop.fit_transform(df['Crop'])
df['State'] = le_state.fit_transform(df['State'])
df['Season'] = le_season.fit_transform(df['Season'])

joblib.dump({
    'Crop': le_crop,
    'State': le_state,
    'Season': le_season
}, 'label_encoders.pkl')

# 5. Train Model (Gradient Boosting Only)
print("⚙️  Training GradientBoostingRegressor...")
feature_cols = ['Crop', 'State', 'Season', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
X = df[feature_cols]
y = df['Yield']

# Train on everything (for production) or split (for validation)
# Here we split just to show the score, but save the trained model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"✅ Training Complete! Model Accuracy (R²): {score*100:.2f}%")

# 6. Save the Best Model
# We save it as a tuple (model, feature_names) because your snippet implied it might be used that way,
# but usually saving just the model object is safer. I will save just the model object.
joblib.dump(model, 'best_model_augmented.pkl')

print("🎉 All files saved: 'best_model_augmented.pkl', 'label_encoders.pkl', etc.")
