def test_health(client):
    response = client.get("/health/")

    assert response.status_code == 200

    assert response.json() == {"status": "ok"}


def test_predict(client, passenger):
    response = client.post("/predict/",
                           json=passenger)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "shap" in data

    assert data["prediction"] in [0, 1]
    assert (0 <= data["probability"] <= 1)


def test_invalid_passenger(client, passenger):
    passenger["pclass"] = 5

    response = client.post("/predict/",
                           json=passenger)

    assert response.status_code == 422

    data = response.json()

    assert data["message"] == ("Validation failed")



