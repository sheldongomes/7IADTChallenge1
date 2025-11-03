import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pathlib import Path

# === CONFIGURAÇÕES ===
MODEL_LOG_PATH = Path("models/logistic_regression.pkl")
MODEL_SVM_PATH = Path("models/svm.pkl")
SCALER_PATH = Path("models/scaler.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")
CSV_PATH = Path("data/patients_test.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# === CARREGAR MODELOS E DADOS ===
print("Carregando modelos e dados...")
log_model = joblib.load(MODEL_LOG_PATH)
svm_model = joblib.load(MODEL_SVM_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

print(f"Logistic: {type(log_model).__name__}")
print(f"SVM: {type(svm_model).__name__}")
print(f"Features: {len(feature_names)}\n")

# === LER CSV ===
df = pd.read_csv(CSV_PATH)
y_real = df['type'].map(lambda x: 1 if 'MALIGNANT' in x.upper() else 0)
X_raw = df.drop(columns=['type']) if 'type' in df.columns else df
X_raw = X_raw[feature_names]  # Forçar ordem
X_scaled = scaler.transform(X_raw)

# === PREDIÇÕES ===
print("Fazendo predições...")
prob_log = log_model.predict_proba(X_scaled)[:, 1]
prob_svm = svm_model.predict_proba(X_scaled)[:, 1]
pred_log = (prob_log >= 0.5).astype(int)
pred_svm = (prob_svm >= 0.5).astype(int)

# === CURVA DE CALIBRAÇÃO ===
print("Gerando curva de calibração...")
fraction_of_positives_log, mean_predicted_value_log = calibration_curve(
    y_real, prob_log, n_bins=5
)
fraction_of_positives_svm, mean_predicted_value_svm = calibration_curve(
    y_real, prob_svm, n_bins=5
)

# === GRÁFICO ===
plt.figure(figsize=(10, 7))
plt.plot(mean_predicted_value_log, fraction_of_positives_log, "s-", label="Logistic Regression", color="#1f77b4")
plt.plot(mean_predicted_value_svm, fraction_of_positives_svm, "o-", label="SVM", color="#ff7f0e")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfeitamente calibrado")
plt.title("Curva de Calibração: Logistic vs SVM", fontsize=14)
plt.xlabel("Probabilidade Prevista (Média por Bin)")
plt.ylabel("Fração de Positivos (Real)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "calibration_curve.png", dpi=200, bbox_inches='tight')
plt.show()

# === TABELA COMPARATIVA ===
results = []
for i in range(len(df)):
    results.append({
        "Paciente": df.iloc[i]['type'],
        "Prob Logistic": f"{prob_log[i]:.1%}",
        "Pred Logistic": "MALIGNANT" if pred_log[i] == 1 else "BENIGN",
        "Prob SVM": f"{prob_svm[i]:.1%}",
        "Pred SVM": "MALIGNANT" if pred_svm[i] == 1 else "BENIGN",
        "Real": "MALIGNANT" if y_real[i] == 1 else "BENIGN"
    })

df_results = pd.DataFrame(results)
print("\n" + "="*100)
print("        COMPARAÇÃO DE PROBABILIDADES: LOGISTIC vs SVM")
print("="*100)
print(df_results.to_string(index=False))
print("="*100)

# === SALVAR TABELA ===
df_results.to_csv(RESULTS_DIR / "comparacao_probabilidades.csv", index=False)
print(f"\nResultados salvos em: {RESULTS_DIR}")