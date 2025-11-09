import pandas as pd, sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler

conn = sqlite3.connect("feature_store/feature_store.db")
df = pd.read_sql_query("SELECT * FROM diabetes_features", conn)
conn.close()

df = df.fillna(df.median(numeric_only=True))
for col in ["BMI_Category","Age_Group"]:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

num_cols = [c for c in df.select_dtypes(include=["int64","float64"]).columns if c!="Outcome"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])
df.drop(['Age','BMI'], axis = 1, inplace = True)

df.to_csv("data/diabetes_preprocessed.csv", index=False)
print("✅ Preprocessed data saved:", df.shape)
