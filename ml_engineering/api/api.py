from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
from joblib import load
import os

model_relative_filepath = '..' + os.path.sep + '..' + os.path.sep + 'model' + os.path.sep + 'heart_failure_clf.joblib'
model_filepath = os.path.join(os.path.dirname(__file__), model_relative_filepath)
clf = load(model_filepath)


class Features(BaseModel):
    age: float
    ejection_fraction: float
    serum_sodium: float
    serum_creatinine: float
    time: float


class PredictRequest(BaseModel):
    features: Features


class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


def get_prediction(request: PredictRequest) -> ModelResponse:
    x = [[request.features.age,
          request.features.ejection_fraction,
          request.features.serum_sodium,
          request.features.serum_creatinine,
          request.features.time]]

    prediction = clf.predict(x)[0]

    prob_false = clf.predict_proba(x)[0][0]
    prob_true = clf.predict_proba(x)[0][1]

    response = ModelResponse(
        predictions=[
            {
                'prediction': prediction,
                'probability_true': prob_true,
                'probability_false': prob_false
            }
        ],
    )
    return response


app = FastAPI()


@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(error="This is a test endpoint.")


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:
    pred = get_prediction(request)
    return pred

