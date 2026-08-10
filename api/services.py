import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

pipeline = joblib.load(BASE_DIR / "xgboost_pipeline.pkl")


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["family_size"] = df["sibsp"] + df["parch"] + 1

    df["isChild"] = (df["age"] < 18).astype(int)

    df["fare_per_person"] = df["fare"] / df["family_size"]

    return df


def predict(passenger: dict) -> dict:
    df = pd.DataFrame([passenger])

    df = feature_engineering(df)

    prediction = pipeline.predict(df)[0]

    probability = pipeline.predict_proba(df)[0][1]

    return {"prediction": int(prediction),
            "probability": float(probability)}