import shap
import pandas as pd
from .services import pipeline, feature_engineering

xgb_model = pipeline.named_steps["model"]

preprocessor = pipeline.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(xgb_model)

def explain(passenger: dict) -> dict:

    df = pd.DataFrame([passenger])

    df = feature_engineering(df)

    processed_data = preprocessor.transform(df)

    shap_values = explainer(processed_data)

    return {
        "base_value": float(shap_values.base_values[0]),
        "shap_values": dict(zip(feature_names, shap_values.values[0]))
    }