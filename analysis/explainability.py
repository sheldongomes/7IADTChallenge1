import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from src.pre_process import load_and_preprocess

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Path to save the results
RESULTS_DIR = Path("results/explainability")
RESULTS_DIR.mkdir(exist_ok=True)

# Loading model
model = joblib.load('models/logistic_regression.pkl')
X_train, X_test, y_train, y_test, _, features = load_and_preprocess('data/breast_cancer_data.csv')

# 1. Feature Importance
coefs = model.coef_[0]
importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefs
}).sort_values('Coefficient', key=abs, ascending=False)

top10 = importance.head(10)
plt.figure(figsize=(10,6))
colors = ['red' if x < 0 else 'blue' for x in top10['Coefficient']]
plt.barh(top10['Feature'][::-1], top10['Coefficient'][::-1], color=colors[::-1])
plt.title('Top 10 Features (Logistic Regression)')
plt.xlabel('Coeficiente')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=150)
plt.close()

# 2. SHAP
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'shap_summary.png', dpi=150)
plt.close()

# 3. Local explanation
idx = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    X_test[idx],
    feature_names=features,
    matplotlib=True,
    show=False
)
plt.title('Local explanation - Patient MALIGNANT')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'shap_force_malignant.png', dpi=150)
plt.close()

print(f"Interpretation completed! Graphics available at: {RESULTS_DIR}")