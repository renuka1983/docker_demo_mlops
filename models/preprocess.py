import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


data = pd.read_csv("data/diabetes.csv")

data = data.fillna(data.median(numeric_only = True))

num_cols = data.select_dtypes(include = ['float64', 'int64'])

if "Outcome" is num_cols:
    num_cols.remove("Outcome")

scaler = StandardScaler()

data[num_cols] = scaler.fit_transform(data[num_cols] )


data.to_csv("data/diabetes_preprocessed.csv", index = False)

print("Preprocessed data has been saved to data/diabetes_preprocessed.csv")

