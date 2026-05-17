import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("heart.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("target", axis=1)

y = df["target"]

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test_scaled)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# =========================
# CREATE MODELS FOLDER
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# SAVE MODEL
# =========================

pickle.dump(
    model,
    open("models/heart_model.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("\nModel Saved Successfully")