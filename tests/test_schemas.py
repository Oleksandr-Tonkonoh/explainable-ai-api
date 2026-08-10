import pytest
from api.schemas import Passenger
from pydantic import ValidationError

def test_valid_passenger(passenger):

    passenger_test = Passenger(**passenger)

    assert passenger_test.pclass == passenger["pclass"]
    assert passenger_test.age == passenger["age"]
    assert passenger_test.sex == passenger["sex"]


def test_invalid_pclass(passenger):

    passenger["pclass"] = 5

    with pytest.raises(ValidationError):
        Passenger(**passenger)