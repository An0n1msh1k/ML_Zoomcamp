import requests

URL = "http://localhost:9696/predict"

payload = {
    "age": 20,
    "gender": "Male",
    "study_hours_per_day": 4.0,
    "social_media_hours": 1.5,
    "netflix_hours": 0.5,
    "part_time_job": "No",
    "attendance_percentage": 92.0,
    "sleep_hours": 7.5,
    "diet_quality": "Good",
    "exercise_frequency": 4,
    "parental_education_level": "Bachelor",
    "internet_quality": "Good",
    "mental_health_rating": 8,
    "extracurricular_participation": "Yes",
}

response = requests.post(URL, json=payload)
print(response.status_code)
print(response.json())