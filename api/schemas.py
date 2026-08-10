from pydantic import BaseModel, Field
from typing import Literal


class Passenger(BaseModel):

    pclass: int = Field(ge=1, le=3)
    age: float = Field(ge=0)
    sibsp: int = Field(ge=0)
    parch: int = Field(ge=0)
    fare: float = Field(ge=0)

    sex: Literal["male", "female"]
    embarked: Literal["S", "C", "Q"]

    adult_male: bool
    alone: bool

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    prediction: int

    probability: float = Field(ge=0, le=1)

    shap: dict[str, float] | None = None