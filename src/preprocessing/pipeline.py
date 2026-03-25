from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Liste des features numériques et catégorielles.
    """
    numeric_features = [
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "hours_since_maintenance",
        "ambient_temp",
        "temp_gap",
        "vibration_rpm_interaction",
        "pressure_current_interaction",
        "vibration_delta",
        "temperature_delta",
        "anomaly_trend_raw",
    ]

    categorical_features = [
        "machine_type",
        "operating_mode",
    ]

    return numeric_features, categorical_features


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Pipeline sklearn complet :
    - numériques : imputation médiane + standardisation
    - catégorielles : imputation mode + one hot encoding
    """
    numeric_features, categorical_features = get_feature_lists()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor