import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Any

PARENTAL_EDU_ORDER = ["High School", "Some College", "Bachelor's", "Master's", "Doctorate"]

NUMERICAL_COLS = [
    "hours_studied",
    "attendance_rate",
    "previous_score",
    "sleep_hours",
    "tutoring_sessions",
    "stress_level",
    "study_efficiency",
    "attendance_risk",
    "academic_momentum",
    "wellbeing_ratio"
]

ORDINAL_COLS = ["parental_education"]

BINARY_CAT_COLS = [
    "internet_access",
    "extracurricular_activities",
    "peer_study_group"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates informative interaction features capturing domain-specific
    academic and lifestyle balance dynamics.
    """
    data = df.copy()

    # 1. Study Efficiency: ratio of study hours to sleep baseline
    data["study_efficiency"] = data["hours_studied"] / (data["sleep_hours"] + 1e-4)

    # 2. Attendance Risk: low attendance penalty threshold (< 75%)
    data["attendance_risk"] = (data["attendance_rate"] < 75.0).astype(float)

    # 3. Academic Momentum: prior baseline combined with active tutoring support
    data["academic_momentum"] = data["previous_score"] + (data["tutoring_sessions"] * 2.5)

    # 4. Wellbeing Balance: sleep hours relative to stress level
    data["wellbeing_ratio"] = data["sleep_hours"] / (data["stress_level"] + 1e-4)

    return data

def build_preprocessor() -> ColumnTransformer:
    """
    Builds a robust scikit-learn ColumnTransformer for numerical scaling
    and categorical encoding.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ord_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[PARENTAL_EDU_ORDER]))
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_COLS),
            ("ord", ord_transformer, ORDINAL_COLS),
            ("cat", cat_transformer, BINARY_CAT_COLS)
        ],
        remainder="drop"
    )

    return preprocessor

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]
]:
    """
    Preprocesses raw student dataframe, engineers features, fits the transformer,
    and returns train/test splits along with transformed feature names.
    """
    # Feature engineering
    engineered_df = engineer_features(df)

    # Drop ID and tier columns if present
    features_to_drop = [c for c in ["student_id", "performance_tier", "final_score"] if c in engineered_df.columns]
    X = engineered_df.drop(columns=features_to_drop)
    y = engineered_df["final_score"].values

    # Train / Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Preprocessor
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(BINARY_CAT_COLS).tolist()
    feature_names = NUMERICAL_COLS + ORDINAL_COLS + cat_names

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, feature_names
