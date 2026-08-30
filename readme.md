# Telco Customer Churn Predictor

A production-style machine learning portfolio project that predicts whether a telecom customer is likely to churn using a trained XGBoost model, exposed through a FastAPI backend and an interactive Gradio interface.

## Project overview

This project demonstrates a complete ML deployment workflow:

- Data preprocessing and feature engineering
- Model training and experimentation tracking
- API deployment with FastAPI
- Interactive UI for business users with Gradio
- Model inference pipeline and validation tests
- Clean, portfolio-ready project structure

## Business use case

Telecom companies need to identify customers at risk of leaving so they can trigger retention actions earlier. This project predicts churn risk based on customer attributes such as contract type, billing behavior, internet service, tenure, and monthly spend.

## Tech stack

- Python 3.11
- FastAPI
- Gradio
- XGBoost
- scikit-learn
- pandas
- joblib
- MLflow
- pytest

## Repository structure

```text
.
├── src/
│   ├── app/
│   │   └── main.py              # FastAPI + Gradio application
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── evaluate.py
│   ├── serving/
│   │   └── inference.py         # Model inference logic
│   └── utils/
│       └── validate_data.py
├── scripts/
│   └── run_pipeline.py          # End-to-end training pipeline
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── artifacts/
├── data/
├── configs/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── readme.md
└── .github/workflows/ci.yml
```

## Setup

### 1) Clone the repository

```bash
git clone https://github.com/your-username/telco-customer-churn-ml.git
cd telco-customer-churn-ml
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# or on Windows:
# .venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Train the model

```bash
python scripts/run_pipeline.py --input data/raw/chum.csv
```

This generates the model artifacts used by the API.

## Run the app

Start the API and UI:

```bash
uvicorn src.app.main:app --reload
```

Then open:

- API docs: http://127.0.0.1:8000/docs
- Gradio UI: http://127.0.0.1:8000/ui

## API example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 85.0
  }'
```

Example response:

```json
{
  "prediction": {
    "label": "Likely to churn",
    "churn_probability": 0.9521
  }
}
```

## Portfolio value

This project highlights:

- end-to-end ML workflow
- production-friendly API architecture
- user-facing model deployment
- clean project organization
- unit-tested ML logic
- GitHub-ready documentation

## License

This project is intended for portfolio and educational use.
