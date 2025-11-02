import joblib
import pandas as pd
from pathlib import Path

# Declaring path for Model and Scaler
MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# Loading Model and Scaler
print("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"Model Loaded: {type(model).__name__}")
print(f"Waiting {model.n_features_in_} features\n")

# New pacient to validate the model
# These are values for a MALIGNANT case (to demonstrate)
new_patient = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.80,
    'area_mean': 1001.0,
    'smoothness_mean': 0.11840,
    'compactness_mean': 0.27760,
    'concavity_mean': 0.3001,
    'concave points_mean': 0.14710,
    'symmetry_mean': 0.2419,
    'fractal_dimension_mean': 0.07871,
    'radius_se': 1.0950,
    'texture_se': 0.9053,
    'perimeter_se': 8.589,
    'area_se': 153.40,
    'smoothness_se': 0.006399,
    'compactness_se': 0.04904,
    'concavity_se': 0.05373,
    'concave points_se': 0.01587,
    'symmetry_se': 0.03003,
    'fractal_dimension_se': 0.006193,
    'radius_worst': 25.38,
    'texture_worst': 17.33,
    'perimeter_worst': 184.60,
    'area_worst': 2019.0,
    'smoothness_worst': 0.1622,
    'compactness_worst': 0.6656,
    'concavity_worst': 0.7119,
    'concave points_worst': 0.2654,
    'symmetry_worst': 0.4601,
    'fractal_dimension_worst': 0.11890
}

# Converting to the same data frame format of the training
X_new = pd.DataFrame([new_patient])

# Scaling the data
X_new_scaled = scaler.transform(X_new)

# Doing the prediction
pred = model.predict(X_new_scaled)[0]
probs = model.predict_proba(X_new_scaled)[0]
prob_malignant = probs[1]
prob_benign = probs[0]

# Displaying results
print("="*60)
print("        TESTING MODEL IN NEW PATIENT")
print("="*60)
print("Patient smaple (original values):")
for k, v in list(new_patient.items())[:6]:
    print(f"  {k:20}: {v:.4f}")
print("  ...")

print(f"\nDiagnostic predicted: {'MALIGNANT' if pred == 1 else 'BENIGN'}")
print(f"  → Confiança MALIGNANT: {prob_malignant:.1%}")
print(f"  → Confiança BENIGN: {prob_benign:.1%}")
print("="*60)