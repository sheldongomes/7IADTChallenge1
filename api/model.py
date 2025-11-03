import joblib
import pandas as pd
from pathlib import Path

# Declaring files path
ROOT_DIR = Path(__file__).parent.parent
BEST_MODEL_PATH = ROOT_DIR / "models" / "logistic_regression.pkl"
RANDOM_MODEL_PATH = ROOT_DIR / "models" / "random_forest.pkl"
SVM_MODEL_PATH = ROOT_DIR / "models" / "svm.pkl"
SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "feature_names.pkl"

# Loading scaler and features
print("Loading Models...")
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
print(f"Waiting {len(features)} features")

def predict_paciente(paciente_dict: dict, model_name):
    #Loading model based on the request path
    print('Model Name: ' + str(model_name))
    if model_name is None or model_name == 'best' or model_name == 'logistic_regression':
        model = joblib.load(BEST_MODEL_PATH)
        print('Case 1')
    elif model_name == 'random_forest':
        model = joblib.load(RANDOM_MODEL_PATH)
        print('Case 2')
    elif model_name == 'svm':
        model = joblib.load(SVM_MODEL_PATH)
        print('Case 3')
    print(f"Model requested: {type(model).__name__}")
    
    # Converting to DataFrame
    X = pd.DataFrame([paciente_dict])
    X = X[features]  # Ensure we have feature sorted
    X_scaled = scaler.transform(X)
    
    # Prediction Benign vs Malignant
    pred = int(model.predict(X_scaled)[0])
    prob = model.predict_proba(X_scaled)[0]
    
    return {
        "model": type(model).__name__,
        "diagnostic": "MALIGNANT" if pred == 1 else "BENIGN",
        "malignant_probability": f"{prob[1]*100:.4f} %",
        "benign_probability": f"{prob[0]*100:.4f} %"
    }