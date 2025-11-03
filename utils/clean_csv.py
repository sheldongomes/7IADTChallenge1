# scripts/clean_csv.py
import pandas as pd
from pathlib import Path

# Caminhos
RAW_PATH = Path("data/breast_cancer_data_raw.csv")
CLEAN_PATH = Path("data/breast_cancer_data.csv")

# 1. Ler o CSV original
print("Lendo CSV original...")
df = pd.read_csv(RAW_PATH)

print(f"Shape original: {df.shape}")
print(f"Colunas originais: {list(df.columns)}")

# 2. Remover colunas completamente vazias (NaN ou Unnamed)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove "Unnamed: 0", etc.
df = df.dropna(axis=1, how='all')  # Remove colunas 100% NaN

print(f"Shape após remover vazias: {df.shape}")

# 3. Verificar se sobrou 32 colunas (id + diagnosis + 30 features)
expected_cols = 32
if df.shape[1] != expected_cols:
    print(f"ERRO: Esperado {expected_cols} colunas, mas tem {df.shape[1]}")
    print("Colunas atuais:", list(df.columns)[:10], "...")
    raise ValueError(f"Verifique o CSV! Colunas: {df.shape[1]}")

# 4. Definir nomes corretos (32 colunas)
column_names = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# 5. Atribuir nomes
df.columns = column_names

# 6. Garantir tipo
df['diagnosis'] = df['diagnosis'].astype(str).str.strip()

# 7. Salvar CSV limpo
df.to_csv(CLEAN_PATH, index=False)
print(f"\nCSV limpo salvo em: {CLEAN_PATH}")
print(f"Colunas finais: {list(df.columns)}")
print(f"Primeiras linhas:\n{df.head(2)}")