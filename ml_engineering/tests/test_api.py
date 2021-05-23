from fastapi.testclient import TestClient

from api.api import app
from api.api import PredictRequest
from api.api import Features
from api.api import get_prediction
from api.api import ModelResponse
import unittest
import json

client = TestClient(app)


class TestGetPrediction(unittest.TestCase):
    def test_model_prediction(self):
        """
        Test that the models prediction successfully returns the expected json
        """
        features = Features(
            age=1,
            ejection_fraction=1,
            serum_sodium=1,
            serum_creatinine=1,
            time=1
        )
        prediction_request = PredictRequest(
            features=features
        )

        json_result = get_prediction(prediction_request)
        json_dict = json.loads(json_result.json())

        prediction_response_json_checker(self, json_dict)


class TestPredictRequest(unittest.TestCase):

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == ModelResponse(error="This is a test endpoint.")

    def test_get_prediction(self):
        response = client.get("/predict")
        assert response.status_code == 200
        assert response.json() == ModelResponse(error="Send a POST request to this endpoint with 'features' data.")

    def test_post_prediction(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 200
        prediction_response_json_checker(self, response.json())

    def test_prediction_with_missing_feature_age(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'age'], 'msg': 'field required',
                                               'type': 'value_error.missing'}]}

    def test_prediction_with_invalid_feature_age(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": "Hello",
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'age'],
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}

    def test_prediction_with_missing_feature_ejection_fraction(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'ejection_fraction'], 'msg': 'field required',
                                               'type': 'value_error.missing'}]}

    def test_prediction_with_invalid_ejection_fraction(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": "Hello",
                    "serum_sodium": 1,
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'ejection_fraction'],
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}

    def test_prediction_with_missing_feature_serum_sodium(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'serum_sodium'], 'msg': 'field required',
                                               'type': 'value_error.missing'}]}

    def test_prediction_with_invalid_serum_sodium(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_sodium": "Hello",
                    "serum_creatinine": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'serum_sodium'],
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}

    def test_prediction_with_missing_feature_serum_creatinine(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'serum_creatinine'], 'msg': 'field required',
                                               'type': 'value_error.missing'}]}

    def test_prediction_with_invalid_serum_creatinine(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": "Hello",
                    "time": 5
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'serum_creatinine'],
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}

    def test_prediction_with_missing_feature_time(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": 1
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'time'], 'msg': 'field required',
                                               'type': 'value_error.missing'}]}

    def test_prediction_with_invalid_feature_time(self):
        response = client.post(
            "/predict",
            json={
                "features": {
                    "age": 1,
                    "ejection_fraction": 1,
                    "serum_sodium": 1,
                    "serum_creatinine": 1,
                    "time": "Hello"
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'features', 'time'],
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}


def prediction_response_json_checker(self, json_dict):

    self.assertIn("predictions", json_dict)
    self.assertIn("prediction", json_dict["predictions"][0])
    self.assertTrue(isinstance(json_dict["predictions"][0]["prediction"], float))
    self.assertIn("probability_true", json_dict["predictions"][0])
    self.assertTrue(isinstance(json_dict["predictions"][0]["probability_true"], float))
    self.assertIn("probability_false", json_dict["predictions"][0])
    self.assertTrue(isinstance(json_dict["predictions"][0]["probability_false"], float))


if __name__ == '__main__':
    unittest.main()
