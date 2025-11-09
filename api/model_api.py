from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np

app = FastAPI()
model = pickle.load(open("models/log_model.sav","rb"))

class Input(BaseModel):
    Pregnancies:int; Glucose:int; BloodPressure:int; SkinThickness:int
    Insulin:int; DiabetesPedigreeFunction:float; BMI_Category:int;Age_Group:int ; 

@app.post("/diabetes_prediction")
def predict(inp: Input):
    arr = np.array([[inp.Pregnancies, inp.Glucose, inp.BloodPressure,
                     inp.SkinThickness, inp.Insulin,
                     inp.DiabetesPedigreeFunction,inp.BMI_Category, inp.Age_Group]])
    y = model.predict(arr)[0]
    return {"prediction": "Patient is diabetic" if y==1 else "Patient is not diabetic"}
