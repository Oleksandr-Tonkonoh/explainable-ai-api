import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def client():

    return TestClient(app)


@pytest.fixture
def passenger():

    return {
        "pclass": 1,
        "age": 28,
        "sibsp": 0,
        "parch": 0,
        "fare": 72.5,
        "sex": "female",
        "embarked": "S",
        "adult_male": False,
        "alone": True
    }