import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


data = pd.read_csv("data/diabetes.csv")

data = data.fillna(data.median(numeric_only = True))


data.to_csv("data/diabetes_preprocessed.csv", index = False)

print("Preprocessed data has been saved to data/diabetes_preprocessed.csv")

