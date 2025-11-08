import pandas as pd, sqlite3, yaml
from pathlib import Path

df = pd.read_csv("data/diabetes.csv")
df["BMI_Category"] = pd.cut(df["BMI"], [0,18.5,25,30,100],
                            labels=["Underweight","Normal","Overweight","Obese"])
df["Age_Group"] = pd.cut(df["Age"], [0,30,50,100],
                         labels=["Young","Middle","Senior"])

conn = sqlite3.connect("feature_store/feature_store.db")
df.to_sql("diabetes_features", conn, if_exists="replace", index=False)
conn.close()

meta = {"dataset":"diabetes_features","features":list(df.columns),"num_records":len(df)}
Path("feature_store/features.yaml").write_text(yaml.dump(meta))
print("✅ Feature store created")
