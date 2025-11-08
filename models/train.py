import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle, mlflow, mlflow.sklearn


mlflow.set_tracking_uri("file:./models/mlruns")
mlflow.set_experiment("diabetes_experiment")

data = pd.read_csv("data/diabetes_preprocessed.csv")

X = data.drop("Outcome", axis = 1)
y = data['Outcome']

model = LogisticRegression()

model.fit(X,y)

acc = accuracy_score(y, model.predict(X))

with mlflow.start_run():
    mlflow.log_metric("Accuracy",acc)
    mlflow.sklearn.log_model(model, "model")


pickle.dump(model, open("models/log_model.sav", 'wb'))