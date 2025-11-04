import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report
from src.pre_process import load_and_preprocess

#Setup of directories
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

#Loading data
print("Loading Data...")
X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess('data/breast_cancer_data.csv')
print(f"Training: {X_train.shape[0]} | Testing: {X_test.shape[0]} | Features: {len(features)}")

# Defining models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Training and Assesment
results = []
best_model_name = None
best_recall = 0

print("\n" + "="*60)
print("        TRAINING 3 MODELS")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Recall for MALIGNANT
    precision = precision_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 5),
        "Precision (MALIGNANT)": round(precision, 5),
        "Recall (MALIGNANT)": round(recall, 5),
        "F1-Score": round(f1, 5)
    })
    
    # Saving models
    joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")
    
    # Updating best model (prioritizing recall)
    if recall > best_recall:
        best_recall = recall
        best_model_name = name

# Displaying results
df_results = pd.DataFrame(results)
print("\n" + "="*60)
print("        RESULT OF MODELS")
print("="*60)
print(df_results.round(4).to_string(index=False))

# Saving results in CSV file
df_results.to_csv(RESULTS_DIR / "models_comparison.csv", index=False)

# Sample with best model
print(f"\nBest Model (greather recall): {best_model_name}")
best_model = joblib.load(MODELS_DIR / f"{best_model_name.lower().replace(' ', '_')}.pkl")

# Selecting Malignant patient with high confidence
idx = next((i for i in range(len(y_test)) 
            if y_test.iloc[i] == 1 and 
               best_model.predict_proba(X_test[i:i+1])[0][1] > 0.90), 
           y_test[y_test == 1].index[0])

sample = X_test[idx:idx+1]
pred = best_model.predict(sample)[0]
probs = best_model.predict_proba(sample)[0]
prob_malignant = probs[1]

print("\n" + "="*60)
print("        REAL CASE DEMONSTRATION (MALIGNANT)")
print("="*60)
print(f"REAL Diagnostic: MALIGNANT")
print("Pacient Data (relevants):")
for i, f in enumerate(features[:6]):
    print(f"  {f:20}: {sample[0][i]:.2f}")
print(f"\nPrediction of Model: {'MALIGNANT' if pred == 1 else 'BENIGN'}")
print(f"  → Confidence MALIGNANT: {prob_malignant:.1%}")
print(f"  → Confidence BENIGN: {probs[0]:.1%}")
print("="*60)

# Saving best model to use later for test (or production if needed)
joblib.dump(features, MODELS_DIR / "feature_names.pkl")
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
print(f"\nModel saved at: {MODELS_DIR}/best_model.pkl")