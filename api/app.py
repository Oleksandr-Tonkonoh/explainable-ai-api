from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .schemas import Passenger, PredictionResponse
from .services import predict
from .explain import explain
import logging

logger = logging.getLogger(__name__)

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError):
    errors = []

    for error in exc.errors():
        errors.append({
            "field": error["loc"][-1],
            "message": error["msg"]
        })

    return JSONResponse(status_code=422,
                        content={
                            "message": "Validation failed",
                            "errors": errors
                        })


@app.exception_handler(Exception)
async def general_exception_handler(request: Request,
                                    exc: Exception):
    logger.exception("Unhandled error: %s %s",
                     request.method,
                     request.url)

    return JSONResponse(status_code=500,
                            content={
                                "detail": "Internal server error"
                            })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_passenger(passenger: Passenger):

    try:
        passenger_data = passenger.model_dump()

        result = predict(passenger_data)

        shap_result = explain(passenger_data)

        return {"prediction": result["prediction"],
                "probability": result["probability"],
                "shap": shap_result["shap_values"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


