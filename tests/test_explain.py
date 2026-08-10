from api.explain import explain


def test_shap_explanation(passenger):
    result = explain(passenger)

    assert "base_value" in result
    assert "shap_values" in result

    assert isinstance(result["base_value"], float)
    assert isinstance(result["shap_values"], dict)