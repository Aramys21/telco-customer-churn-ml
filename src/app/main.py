"""Portfolio-ready ML application for telco churn prediction."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

from src.serving.inference import predict

app = FastAPI(
    title="Telco Customer Churn Predictor",
    description=(
        "Machine learning solution for predicting telecom customer churn. "
        "The app exposes a production-ready REST API and a user-friendly demo UI."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/")
def root() -> dict[str, str]:
    """Health check endpoint used by monitoring tools and deployment checks."""
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Portfolio-friendly health endpoint exposing the service state."""
    return {
        "status": "ok",
        "service": "telco-churn-predictor",
        "api_ready": True,
        "ui_ready": True,
    }


# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.

    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """

    # Demographics
    gender: str  # "Male" or "Female"
    Partner: str  # "Yes" or "No" - has partner
    Dependents: str  # "Yes" or "No" - has dependents

    # Phone services
    PhoneService: str  # "Yes" or "No"
    MultipleLines: str  # "Yes", "No", or "No phone service"

    # Internet services
    InternetService: str  # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str  # "Yes", "No", or "No internet service"
    OnlineBackup: str  # "Yes", "No", or "No internet service"
    DeviceProtection: str  # "Yes", "No", or "No internet service"
    TechSupport: str  # "Yes", "No", or "No internet service"
    StreamingTV: str  # "Yes", "No", or "No internet service"
    StreamingMovies: str  # "Yes", "No", or "No internet service"

    # Account information
    Contract: str  # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str  # "Yes" or "No"
    PaymentMethod: str  # "Electronic check", "Mailed check", etc.

    # Numeric features
    tenure: int = Field(ge=0, le=120)  # Number of months with company
    MonthlyCharges: float = Field(ge=0, le=200)  # Monthly charges in dollars
    TotalCharges: float = Field(ge=0, le=100_000)  # Total charges to date


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.

    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format

    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


# =================================================== #


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
):
    """
    Gradio interface function that processes form inputs and returns prediction.

    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string

    """
    # Construct data dictionary matching CustomerData schema
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),  # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges),  # Ensure float type
    }

    # Call same inference pipeline as API endpoint
    try:
        return str(predict(data))
    except Exception as exc:
        return f"Prediction unavailable: {exc}"


# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all customer features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        # Phone services section
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(
            ["Yes", "No", "No phone service"], label="Multiple Lines", value="No"
        ),
        # Internet services section (key churn predictors)
        gr.Dropdown(
            ["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Online Security", value="No"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Online Backup", value="No"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Device Protection", value="No"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Tech Support", value="No"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"
        ),
        # Contract and billing section (major churn factors)
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month",
        ),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
            value="Electronic check",
        ),
        # Numeric features (important for churn prediction)
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="""
    <div style='text-align: left; font-size: 1.0rem; line-height: 1.5;'>
        <strong>Customer churn risk assessment</strong><br>
        This demo uses a trained machine learning model to estimate whether a telecom customer is likely to leave.
        <br><br>
        <em>Portfolio project:</em> operational ML, API deployment, and user-facing analytics in a single application.
    </div>
    """,
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    examples=[
        # High churn risk example
        [
            "Female",
            "No",
            "No",
            "Yes",
            "No",
            "Fiber optic",
            "No",
            "No",
            "No",
            "No",
            "Yes",
            "Yes",
            "Month-to-month",
            "Yes",
            "Electronic check",
            1,
            85.0,
            85.0,
        ],
        # Low churn risk example
        [
            "Male",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "DSL",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "No",
            "Two year",
            "No",
            "Credit card (automatic)",
            60,
            45.0,
            2700.0,
        ],
    ],
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,  # FastAPI application instance
    demo,  # Gradio interface
    path="/ui",  # URL path where Gradio will be accessible
)
