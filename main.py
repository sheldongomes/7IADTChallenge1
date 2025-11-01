import joblib
from sklearn.ensemble import RandomForestClassifier
from src.pre_process import load_and_preprocess

print("=== Diagnostic Support System ===")
#Running pre_process.py with breast cancer Kaggle file from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download
X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess('data/breast_cancer_wisconsin_data.csv')

# Creating simple model with RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'models/rf_model.pkl')

# Prediction example with 1st patient sample data:
sample = X_test[:1]
pred = model.predict(sample)[0]
prob_malignant = model.predict_proba(sample)[:, 1][0]
print("Patient data:")
for i, feature in enumerate(features):
    print(f"  {feature}: {sample[0][i]:.2f}")
print(f"Prediction: {'Malignant' if pred == 1 else 'Benign'} (Malignant Probability: {prob_malignant:.2f})")
print("Model saved at models/rf_model.pkl")