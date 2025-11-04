import pandas as pd
from pathlib import Path

def clean_csv():
    # Declaring data path
    RAW_PATH = Path("data/breast_cancer_data_raw.csv")
    CLEAN_PATH = Path("data/breast_cancer_data.csv")

    # 1. Loading original CSV file
    print("Loading original CSV file...")
    df = pd.read_csv(RAW_PATH)

    print(f"Original shape: {df.shape}")
    print(f"Original Colums: {list(df.columns)}")

    # 2. Removing empty columns (NaN ou Unnamed)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove "Unnamed: 0", etc.
    df = df.dropna(axis=1, how='all')  # Remove columns 100% NaN

    print(f"Shape after empty removed: {df.shape}")

    # 3. Checking if remain 32 columns (id + diagnosis + 30 features)
    expected_cols = 32
    if df.shape[1] != expected_cols:
        print(f"ERROR: Expected {expected_cols} columns, but remain {df.shape[1]}")
        print("Current columns:", list(df.columns)[:10], "...")
        raise ValueError(f"Check CSV file! Columns: {df.shape[1]}")

    # 4. Defining correct names (32 columns)
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

    # 5. Assigning names
    df.columns = column_names

    # 6. Ensuring the type
    df['diagnosis'] = df['diagnosis'].astype(str).str.strip()

    # 7. Writing final CSV (with intended values)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nFinal CSV saved at: {CLEAN_PATH}")
    print(f"Final columns: {list(df.columns)}")
    print(f"First lines:\n{df.head(2)}")