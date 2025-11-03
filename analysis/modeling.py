import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from src.pre_process import load_and_preprocess

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Path to save the results
RESULTS_DIR = Path("results/modeling")
RESULTS_DIR.mkdir(exist_ok=True)

# Loading Data
X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess('data/breast_cancer_data.csv')
print(f"Training: {X_train.shape} | Testing: {X_test.shape}")

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []
trained_models = {}

print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Recall (Malignant)': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4)
    })
    
    trained_models[name] = model
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')

# Table
df_results = pd.DataFrame(results)
print("\Results:")
print(df_results.to_string(index=False))
df_results.to_csv(RESULTS_DIR / "models_comparison.csv", index=False)

# Calibration curve
log_model = trained_models['Logistic Regression']
svm_model = trained_models['SVM']

plt.figure(figsize=(10,7))
for prob, label in [(log_model.predict_proba(X_test)[:,1], 'Logistic'), 
                    (svm_model.predict_proba(X_test)[:,1], 'SVM')]:
    fraction, mean = calibration_curve(y_test, prob, n_bins=5)
    plt.plot(mean, fraction, 's-', label=label)
plt.plot([0,1], [0,1], '--', color='gray')
plt.title('Calibration curve')
plt.legend()
plt.savefig(RESULTS_DIR / 'calibration_curve.png', dpi=150)
plt.close()

print(f"\nModels saved at /models/ | Results in {RESULTS_DIR}")