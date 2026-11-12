import pandas as pd

CATEGORICAL_COLS = [
    "gender",
    "part_time_job",
    "diet_quality",
    "parental_education_level",
    "internet_quality",
    "extracurricular_participation",
]

NUMERICAL_COLS = [
    "age",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
]

def prepare_features(df):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    df["total_screen_time"] = df["social_media_hours"] + df["netflix_hours"]
    df["screen_share"] = df["total_screen_time"] / (
        df["total_screen_time"] + df["study_hours_per_day"] + 1e-5
    )
    return df