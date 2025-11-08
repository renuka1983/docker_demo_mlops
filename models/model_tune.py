import pandas as pd, mlflow, optuna, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./models/mlruns")
mlflow.set_experiment("diabetes_hyperparam_tuning")

df = pd.read_csv("data/diabetes_preprocessed.csv")
X, y = df.drop("Outcome", axis=1), df["Outcome"]

def objective(trial):
    params = {
        "C": trial.suggest_float("C",0.01,10.0,log=True),
        "solver": trial.suggest_categorical("solver",["lbfgs","liblinear"]),
        "max_iter": trial.suggest_int("max_iter",100,500)
    }
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    m = LogisticRegression(**params).fit(Xtr,ytr)
    acc = accuracy_score(yte,m.predict(Xte))
    mlflow.log_params(params); mlflow.log_metric("accuracy", acc)
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best = study.best_params
best_model = LogisticRegression(**best).fit(X,y)
mlflow.sklearn.log_model(best_model,"best_model")
pickle.dump(best_model, open("models/log_model.sav","wb"))
print("✅ Best params:", best)
