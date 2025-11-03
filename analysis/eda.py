import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Exploration Data Analysis
# File path and configuration
RESULTS_DIR = Path("results/eda/")
RESULTS_DIR.mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Loading data
print("Loading dataset...")
df = pd.read_csv('data/breast_cancer_data.csv')
df['diagnosis'] = df['diagnosis'].map({'M': 'Malignant', 'B': 'Benign'})
print(f"Dataset: {df.shape[0]} patients, {df.shape[1]} columns")

# 1. Distribution between Benign vs Malignant
plt.figure(figsize=(8,6))
df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Distribution: Benign vs Malignant')
plt.ylabel('')
plt.savefig(RESULTS_DIR / 'diagnosis_pie.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Descriptive Statistics
print("\nCalculating descriptive statistics...")

# Selecting Features
features = ['radius_mean', 'area_mean', 'concave_points_mean']
stats = df.groupby('diagnosis')[features].mean().round(2)

# Calculating percentage difference
benign = stats.loc['Benign']
malignant = stats.loc['Malignant']
diff_percent = ((malignant - benign) / benign * 100).round(1)

# Building table
stats_final = pd.DataFrame({
    'Benign': benign,
    'Malignant': malignant,
    'Difference (%)': diff_percent
}).T

print("\nDescriptive Statistics:")
print(stats_final)

# Saving in CSV file
stats_final.to_csv(RESULTS_DIR / "desctiptive_statistics.csv")
print(f"\nTable saved at: {RESULTS_DIR / 'desctiptive_statistics.csv'}")

# 3. Histogram
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='area_mean', hue='diagnosis', kde=True, alpha=0.7, palette=['#66b3ff', '#ff9999'])
plt.title('Distribution of Average Area by Diagnosis')
plt.xlabel('Average Area')
plt.savefig(RESULTS_DIR / 'dist_area_mean.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='diagnosis', y='concave_points_worst', palette=['#66b3ff', '#ff9999'])
plt.title('Concave Pints (worst region): Benign vs Malignant')
plt.ylabel('Concave Points Worst')
plt.savefig(RESULTS_DIR / 'box_concave_worst.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Heatmap
plt.figure(figsize=(12,10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig(RESULTS_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nExploration Data Analysis Completed! Graphics stored at: {RESULTS_DIR}")