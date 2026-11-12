import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from features import prepare_features

def train():
    csv_path = next(Path("data").glob("*.csv"))
    df = pd.read_csv(csv_path, keep_default_na=False).drop(columns=["student_id"])

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.pop("exam_score").to_numpy()
    y_test = df_test.pop("exam_score").to_numpy()

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(prepare_features(df_full_train).to_dict(orient="records"))
    X_test = dv.transform(prepare_features(df_test).to_dict(orient="records"))

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_full_train, y_full_train)

    rmse = root_mean_squared_error(y_test, model.predict(X_test).clip(0.0, 100.0))
    print(f"Test RMSE: {rmse:.4f}")

    with open("model.bin", "wb") as f_out:
        pickle.dump((dv, model), f_out)

if __name__ == "__main__":
    train()