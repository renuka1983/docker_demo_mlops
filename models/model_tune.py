import pandas as pd, mlflow, mlflow.sklearn, optuna, pickle, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# MLflow setup
mlflow.set_tracking_uri("file:./models/mlruns")
mlflow.set_experiment("diabetes_hyperparam_tuning")

# Load preprocessed data
df = pd.read_csv("data/diabetes_preprocessed.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

def objective(trial):
    params = {
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        "max_iter": trial.suggest_int("max_iter", 100, 500)
    }

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(**params)
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    
    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)

    # Start a new MLflow run for each trial
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})

    return acc

# Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best = study.best_params
print("✅ Best Hyperparameters:", best)

# Train final model on all data
final_model = LogisticRegression(**best)
final_model.fit(X, y)

# Evaluate final model
y_pred_full = final_model.predict(X)
acc = accuracy_score(y, y_pred_full)
prec = precision_score(y, y_pred_full, zero_division=0)
rec = recall_score(y, y_pred_full, zero_division=0)

# Log best model and metrics to MLflow
with mlflow.start_run(run_name="best_model"):
    mlflow.log_params(best)
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})
    mlflow.sklearn.log_model(final_model, "best_model")

# Save model file
pickle.dump(final_model, open("models/log_model.sav", "wb"))

# ✅ Save metrics for DVC tracking
metrics = {"accuracy": acc, "precision": prec, "recall": rec}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Tuned model saved with metrics:", metrics)
