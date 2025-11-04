import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.clean_csv import clean_csv

def load_and_preprocess(data_path):
    # Load data using pandas. The path of CSV file must be passed to call this function
    if os.path.isfile(data_path):
        print("Formatted file exists")
    else:
        print('Formatting file')
        clean_csv()
    
    file_data = pd.read_csv(data_path)
    file_data = file_data.drop('id', axis=1)
    
    # Diagnosis: Will be classified as: Malignant -> 1, Benign -> 0
    file_data['diagnosis'] = file_data['diagnosis'].map({'M': 1, 'B': 0})
    
    X = file_data.drop('diagnosis', axis=1)
    y = file_data['diagnosis']
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        # The value 0.2 as size means we will use 80% of the data to learn and 20% to test
        # random_state is being used to have equal results in all tests (the value 42 is the magic number from Douglas Adams book, but any value can be used)
        # Stratify=y will keeps the same proportion of Malignant and Benign between test and training data
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the train and test results to be returned
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # return data
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()