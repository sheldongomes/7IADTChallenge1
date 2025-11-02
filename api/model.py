# api/model.py
import joblib
import pandas as pd
from pathlib import Path

# Caminho relativo
ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / "models" / "logistic_regression.pkl"
SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "feature_names.pkl"

# Carregar
print("Carregando modelo...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
print(f"Modelo carregado: {type(model).__name__}")
print(f"Esperando {len(features)} features")

def predict_paciente(paciente_dict: dict):
    # Converter para DataFrame
    X = pd.DataFrame([paciente_dict])
    X = X[features]  # Garantir ordem
    X_scaled = scaler.transform(X)
    
    # Predição
    pred = int(model.predict(X_scaled)[0])
    prob = model.predict_proba(X_scaled)[0]
    
    return {
        "diagnostico": "MALIGNO" if pred == 1 else "BENIGNO",
        "probabilidade_maligno": round(prob[1], 4),
        "probabilidade_benigno": round(prob[0], 4)
    }