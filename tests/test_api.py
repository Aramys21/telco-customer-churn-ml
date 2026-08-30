from fastapi.testclient import TestClient

from src.app import main


PAYLOAD = {
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1,
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0,
}


def test_health_check_returns_ok() -> None:
    response = TestClient(main.app).get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_inference_result(monkeypatch) -> None:
    expected = {"label": "Likely to churn", "churn_probability": 0.9521}
    monkeypatch.setattr(main, "predict", lambda payload: expected)

    response = TestClient(main.app).post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"prediction": expected}


def test_predict_rejects_invalid_customer_data() -> None:
    invalid_payload = {**PAYLOAD, "tenure": -1}

    response = TestClient(main.app).post("/predict", json=invalid_payload)

    assert response.status_code == 422
