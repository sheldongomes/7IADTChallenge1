import joblib
import pandas as pd
from pathlib import Path

# Declaring files path
MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")
CSV_PATH = Path("data/patients_test.csv")

# Loading Model, Scaler and Feature Names
print("Loading model, scaler e feature names...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

print(f"Model: {type(model).__name__}")
print(f"Waiting {len(feature_names)} features\n")

# Loading patients
print(f"Loading patients from: {CSV_PATH}")
file_data = pd.read_csv(CSV_PATH)

# Removing 'type' column that does not exist in learn initial list of features and reordering
if 'type' in file_data.columns:
    y_real = file_data['type']
    X_new = file_data.drop(columns=['type'])
else:
    y_real = None
    X_new = file_data.copy()

# Force to sort it correct
X_new = X_new[feature_names]

# Scale and prediction
X_scaled = scaler.transform(X_new)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)

# Displaying results
print("\n" + "="*80)
print("        CSV FILE TEST RESULTS:")
print("="*80)

for i in range(len(file_data)):
    type_real = y_real.iloc[i] if y_real is not None else "Unknown"
    pred = preds[i]
    prob_malignant = probs[i][1]
    
    print(f"\nPatient {i+1}: {type_real}")
    print(f"  → Prediction: {'MALIGNANT' if pred == 1 else 'BENIGN'}")
    print(f"  → Confidence MALIGNANT: {prob_malignant:.1%}")
    print(f"  → {'ACCURATE' if (pred == 1 and 'MALIGNANT' in type_real.upper()) or (pred == 0 and 'BENIGN' in type_real.upper()) else 'ERROR'}")
    print("  Sample Values:")
    for j, col in enumerate(feature_names[:3]):
        print(f"    {col}: {X_new.iloc[i, j]:.4f}")

print("\n" + "="*80)