# Student Exam Score Prediction

Midterm project for ML Zoomcamp (2026). The goal is to predict a student's final exam score (from 0 to 100) using features like daily study hours, screen time, sleep, and lifestyle habits.

Live demo: https://an0n1msh1k.github.io/ML_Zoomcamp/Midterm_project/

## Problem description

Students spend a lot of time on social media and streaming services, which often affects their academic results. In this project, I explore how daily habits (study time, sleep, social media, netflix, exercise) correlate with exam performance and train regression models to predict the final score.

I tested three models:
- Ridge regression (baseline linear model with L2 penalty)
- Random forest regressor
- XGBoost regressor

The final model is an XGBoost regressor, which gave an RMSE around 5.6 on the test set.

## Dataset

- Source: [Student Habits vs Academic Performance on Kaggle](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance)
- Size: 1,000 rows, 14 input features, 1 target (`exam_score`)
- Main observations from EDA:
  - `study_hours_per_day` has the highest positive correlation with the exam score (~0.83).
  - `social_media_hours` and `netflix_hours` show negative correlation (~ -0.17).
  - There are no missing values in the dataset. Note that `"None"` in `parental_education_level` is a valid category, so the CSV is loaded with `keep_default_na=False`.
  - Added two engineered features: `total_screen_time` (social media + netflix) and `screen_share` (ratio of screen time to study + screen time).

## Model results

Validation results on 20% holdout validation set:

| Model | Parameters | Validation RMSE |
|---|---|---|
| Ridge regression | alpha = 10.0 | 5.74 |
| Random forest | max_depth = 12, n_estimators = 200 | 6.30 |
| XGBoost | learning_rate = 0.05, max_depth = 3, n_estimators = 300 | 5.93 |

Final model evaluation on the test set:
- Test RMSE: ~5.62 (clipped between 0 and 100)

## Project structure

- `notebook.ipynb` - exploratory data analysis, feature engineering, and model tuning
- `features.py` - feature preprocessing functions shared between training and prediction
- `train.py` - script to train the final XGBoost model and save it to `model.bin`
- `predict.py` - FastAPI app exposing `/predict` and `/health` endpoints
- `test_service.py` - small script to test the prediction API locally
- `model.bin` - serialized DictVectorizer and XGBoost model
- `index.html` - frontend for GitHub Pages with in-browser prediction
- `Dockerfile` - container configuration using uv
- `pyproject.toml`, `uv.lock` - project dependencies

## How to run locally

### 1. Using uv

Install dependencies:
```bash
uv sync
```

Train the model:
```bash
uv run python train.py
```

Run the FastAPI web service:
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

In another terminal, test the service:
```bash
uv run python test_service.py
```

### 2. Using Docker

Build the Docker image:
```bash
docker build -t midterm-project .
```

Run the container:
```bash
docker run -p 9696:9696 midterm-project
```

### 3. Web demo

You can also test the model directly in the browser via GitHub Pages:
https://an0n1msh1k.github.io/ML_Zoomcamp/Midterm_project/

## API request example

POST `/predict`

```json
{
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
  "extracurricular_participation": "Yes"
}
```

Response:
```json
{
  "predicted_exam_score": 88.98
}
```
