import pandas as pd

from src.serving.inference import _prepare_features


def test_prepare_features_encodes_and_aligns_saved_schema() -> None:
    payload = {
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "PaperlessBilling": "No",
        "InternetService": "Fiber optic",
        "tenure": 12,
        "ignored_value": "not part of the model schema",
    }
    feature_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "InternetService_Fiber optic",
        "tenure",
        "missing_training_feature",
    ]

    features = _prepare_features(payload, feature_columns)

    assert features.columns.tolist() == feature_columns
    assert features.dtypes.eq(float).all()
    assert features.loc[0, "gender"] == 1.0
    assert features.loc[0, "Partner"] == 1.0
    assert features.loc[0, "Dependents"] == 0.0
    assert features.loc[0, "InternetService_Fiber optic"] == 1.0
    assert features.loc[0, "missing_training_feature"] == 0.0


def test_prepare_features_handles_missing_optional_categories() -> None:
    features = _prepare_features(
        {"gender": "Female", "tenure": 6},
        ["gender", "Partner", "tenure"],
    )

    pd.testing.assert_frame_equal(
        features,
        pd.DataFrame([[0.0, 0.0, 6.0]], columns=["gender", "Partner", "tenure"]),
    )
