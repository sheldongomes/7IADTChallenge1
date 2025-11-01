import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # Load data
    data = pd.read_csv('../data/breast_cancer_wisconsin_data.csv')
    data = data.drop('id', axis=1)
    
    # Diagnosis: Malignant -> 1, Benign -> 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

rX_train_scaled, rX_test_scaled, ry_train, ry_test, rscaler, rX_columns  = load_and_preprocess()
print('Result: ')
print(ry_test)
