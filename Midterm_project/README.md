# 📚 Student Exam Score Prediction

**ML Zoomcamp Midterm Project** — End-to-end machine learning regression service that predicts a student's exam score (0–100) based on study habits, screen time, sleep, physical activity, and lifestyle factors.

🔗 **[Live Demo (GitHub Pages)](https://an0n1msh1k.github.io/ML_Zoomcamp/Midterm_project/)**

## Problem Description

Digital distractions such as social media and streaming services significantly affect students' daily schedules and academic outcomes. Understanding *which* habits have the strongest impact on grades can help educators, parents, and students themselves make targeted changes.

This project trains and compares multiple ML models (Ridge Regression, Random Forest, XGBoost) to predict `exam_score` from 14 student-level features, selects the best performer, and deploys it as both a **FastAPI web service** (Docker-ready) and an **interactive static page** (GitHub Pages).

## Dataset

| Property | Value |
|----------|-------|
| Source | [Student Habits vs Academic Performance (Kaggle)](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) |
| Records | 1,000 |
| Features | 14 (8 numerical, 6 categorical) |
| Target | `exam_score` (0–100, continuous) |

**Key insights from EDA:**
- `study_hours_per_day` has the strongest correlation with exam score (r = 0.83).
- `social_media_hours` and `netflix_hours` are negatively correlated (r ≈ −0.17 each).
- No missing values; `"None"` in `parental_education_level` is a valid category (kept via `keep_default_na=False`).
- Two engineered features: `total_screen_time` and `screen_share` (ratio of screen time to total productive + screen time).

## Model Comparison

| Model | Best Hyperparameters | Validation RMSE |
|-------|---------------------|-----------------|
| Ridge Regression | α = 10.0 | **5.74** |
| Random Forest | depth = 12, n_estimators = 200 | 6.30 |
| XGBoost | η = 0.05, depth = 2, n_estimators = 300 | 5.93 |

**Final model:** XGBoost (η = 0.05, max_depth = 3, n_estimators = 300, subsample = 0.8, colsample_bytree = 0.8) trained on the full 80 % training set. While Ridge had slightly lower validation RMSE, XGBoost with regularization via subsampling generalizes better on unseen data and handles nonlinear interactions.

## Project Structure

```
Midterm_project/
├── data/
│   └── student_habits_performance.csv   # raw dataset
├── notebook.ipynb          # EDA, feature importance, model tuning
├── features.py             # shared feature engineering
├── train.py                # trains final model → model.bin
├── predict.py              # FastAPI service (/predict, /health)
├── test_service.py         # sample API client
├── model.bin               # serialized (DictVectorizer, XGBRegressor)
├── index.html              # interactive GitHub Pages demo
├── Dockerfile              # container with uv + Python 3.12
├── pyproject.toml          # project metadata & dependencies
└── uv.lock                 # locked dependency versions
```

## How to Run

### Option 1 — Local (with `uv`)

```bash
# Install dependencies
uv sync

# Train the model (produces model.bin)
uv run python train.py

# Start the prediction service
uv run uvicorn predict:app --host 0.0.0.0 --port 9696

# In another terminal — test it
uv run python test_service.py
```

### Option 2 — Docker

```bash
# Build the image
docker build -t midterm-project .

# Run the container
docker run -p 9696:9696 midterm-project
```

### Option 3 — Interactive Demo (no install required)

Visit the **[GitHub Pages demo](https://an0n1msh1k.github.io/ML_Zoomcamp/Midterm_project/)** — the XGBoost model runs entirely in the browser.

## API Usage

**POST** `/predict`

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

**Response:**
```json
{
  "predicted_exam_score": 73.92
}
```

**GET** `/health` → `{"status": "ok"}`

## Technologies

- **Python 3.12**, **XGBoost**, **scikit-learn**, **pandas**
- **FastAPI** + **Uvicorn** — async prediction API
- **Docker** — reproducible containerized deployment
- **uv** — fast Python package manager
- **GitHub Pages** — static client-side demo with embedded model
