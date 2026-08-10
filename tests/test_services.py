from api.services import predict, feature_engineering

def test_feature_engineering(passenger):
    import pandas as pd

    df = pd.DataFrame([passenger])

    result = feature_engineering(df)

    assert "family_size" in result.columns
    assert "isChild" in result.columns
    assert "fare_per_person" in result.columns


def test_prediction(passenger):

    result = predict(passenger)

    assert result["prediction"] in [0, 1]
    assert (0 <= result["probability"] <= 1)