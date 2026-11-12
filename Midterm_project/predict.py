import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from features import prepare_features

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

class StudentHabits(BaseModel):
    age: int
    gender: str
    study_hours_per_day: float
    social_media_hours: float
    netflix_hours: float
    part_time_job: str
    attendance_percentage: float
    sleep_hours: float
    diet_quality: str
    exercise_frequency: int
    parental_education_level: str
    internet_quality: str
    mental_health_rating: int
    extracurricular_participation: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_score(student: StudentHabits):
    df_input = pd.DataFrame([student.model_dump()])
    X = dv.transform(prepare_features(df_input).to_dict(orient="records"))
    pred = round(max(0.0, min(100.0, float(model.predict(X)[0]))), 2)
    return {"predicted_exam_score": pred}