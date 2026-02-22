import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("../data/traffic_dataset.csv")

# Feature Engineering
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['is_rush_hour'] = df['hour'].apply(
    lambda h: 1 if (7<=h<=10 or 17<=h<=20) else 0
)

# Features and Target
features = [
    'hour','day_of_week','month','weather_encoded',
    'vehicle_count','incident_encoded','is_holiday',
    'hour_sin','hour_cos','is_weekend','is_rush_hour'
]

X = df[features]
y = df['congestion_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train model
print("Training model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("../backend/models", exist_ok=True)
with open("../backend/models/ml_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../backend/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model saved to backend/models/")