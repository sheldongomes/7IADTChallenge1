# 7IADTChallenge1
1st FIAP AI tech challenge

Code Description:

__Breast Cancer Data:__ This code was developed using Kaggle breast cancer database.
The source of data used on this project is available at: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data The file is saved at: '/data/breast_cancer_data_raw.csv'

__Project CSV data (mofified):__ During the development, tests and execution I had some issues with the original format.
I decided to create the code /utils/clean_csv.py to execute two changes in the file:
* Delete the last column of the file that did not have any data (NaN values).
* Replace " " space character by "_" underline in file column names. For example, "concave points_mean" was replaced by "concave_points_mean".
These changes were helpfull to keep a patter in feature names and usefull when testing the model locally or through the API code I created to expose it.


Breast Cancer data downloaded from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download
#NOTE: This file contains one more column without value. We deleted this column in the file we are using in this code
