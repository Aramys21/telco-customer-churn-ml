import httpx
import pytest

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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_check_returns_ok() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://testserver"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_predict_returns_inference_result(monkeypatch) -> None:
    expected = {"label": "Likely to churn", "churn_probability": 0.9521}
    monkeypatch.setattr(main, "predict", lambda payload: expected)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://testserver"
    ) as client:
        response = await client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"prediction": expected}


@pytest.mark.anyio
async def test_predict_rejects_invalid_customer_data() -> None:
    invalid_payload = {**PAYLOAD, "tenure": -1}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://testserver"
    ) as client:
        response = await client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
