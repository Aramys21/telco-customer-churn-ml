"""Data-quality checks for the Telco customer churn dataset."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "customerID", "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
}


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate schema, accepted categories, and numeric business ranges."""
    failures: List[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        return False, [f"missing required columns: {', '.join(missing)}"]

    if df["customerID"].isna().any() or (df["customerID"].astype(str).str.strip() == "").any():
        failures.append("customerID contains missing values")

    allowed_values = {
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "Contract": {"Month-to-month", "One year", "Two year"},
    }
    for column, allowed in allowed_values.items():
        invalid = sorted(set(df[column].dropna().astype(str).str.strip()) - allowed)
        if invalid:
            failures.append(f"{column} has invalid values: {invalid}")

    numeric_ranges = {"tenure": (0, 120), "MonthlyCharges": (0, 200), "TotalCharges": (0, None)}
    for column, (minimum, maximum) in numeric_ranges.items():
        raw_values = df[column].astype(str).str.strip()
        values = pd.to_numeric(raw_values, errors="coerce")
        # The public Telco source has blank TotalCharges for new customers;
        # preprocessing intentionally converts these to zero.
        invalid_values = values.isna() & raw_values.ne("")
        if invalid_values.any():
            failures.append(f"{column} contains missing or non-numeric values")
        elif (values < minimum).any() or (maximum is not None and (values > maximum).any()):
            failures.append(f"{column} is outside its allowed range")

    return not failures, failures
