import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables dérivées simples utiles pour la modélisation supervisée.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if {"temperature_motor", "ambient_temp"}.issubset(df.columns):
        df["temp_gap"] = df["temperature_motor"] - df["ambient_temp"]

    if {"vibration_rms", "rpm"}.issubset(df.columns):
        df["vibration_rpm_interaction"] = df["vibration_rms"] * df["rpm"]

    if {"pressure_level", "current_phase_avg"}.issubset(df.columns):
        df["pressure_current_interaction"] = (
            df["pressure_level"] * df["current_phase_avg"]
        )

    if {"machine_id", "timestamp", "vibration_rms", "temperature_motor"}.issubset(df.columns):
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

        df["vibration_delta"] = df.groupby("machine_id")["vibration_rms"].diff().fillna(0)
        df["temperature_delta"] = df.groupby("machine_id")["temperature_motor"].diff().fillna(0)

        df["anomaly_trend_raw"] = (
            0.6 * df["vibration_delta"].clip(lower=0)
            + 0.4 * df["temperature_delta"].clip(lower=0)
        )

    return df


def robust_minmax_normalize(values: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """
    Normalisation robuste entre 0 et 1 basée sur les quantiles.
    """
    low = np.quantile(values, lower_q)
    high = np.quantile(values, upper_q)

    if high - low < 1e-12:
        return np.zeros_like(values)

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def build_autoencoder(input_dim: int, latent_dim_1: int = 16, latent_dim_2: int = 8) -> Model:
    """
    Construit un autoencodeur dense simple.
    """
    inputs = Input(shape=(input_dim,), name="input_layer")

    x = Dense(latent_dim_1, activation="relu", name="encoder_dense_1")(inputs)
    x = Dense(latent_dim_2, activation="relu", name="encoder_dense_2")(x)

    x = Dense(latent_dim_2, activation="relu", name="decoder_dense_1")(x)
    x = Dense(latent_dim_1, activation="relu", name="decoder_dense_2")(x)
    outputs = Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="autoencoder_health_index")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_health_index_ae(
    df: pd.DataFrame,
    sensor_features: list[str] | None = None,
    target_failure_col: str = "failure_within_24h",
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> tuple[pd.DataFrame, Model, SimpleImputer, StandardScaler]:
    """
    Construit un Health Index via autoencodeur entraîné uniquement sur les lignes normales
    (failure_within_24h == 0).

    Retourne :
    - dataframe enrichi
    - autoencodeur entraîné
    - imputer appris sur train normal
    - scaler appris sur train normal
    """
    df = df.copy()

    if sensor_features is None:
        sensor_features = [
            "vibration_rms",
            "temperature_motor",
            "pressure_level",
            "rpm",
        ]

    required_cols = sensor_features + [target_failure_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    X_raw = df[sensor_features].copy()

    normal_mask = df[target_failure_col] == 0
    df_normal = df.loc[normal_mask].copy()

    if df_normal.empty:
        raise ValueError("Aucune ligne avec failure_within_24h == 0 trouvée.")

    X_normal_raw = df_normal[sensor_features].copy()

    X_train_raw, X_val_raw = train_test_split(
        X_normal_raw,
        test_size=0.2,
        random_state=random_state
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)

    X_train = scaler.fit_transform(X_train_imp)
    X_val = scaler.transform(X_val_imp)

    X_all_imp = imputer.transform(X_raw)
    X_all = scaler.transform(X_all_imp)

    autoencoder = build_autoencoder(input_dim=X_train.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    X_all_reconstructed = autoencoder.predict(X_all, verbose=0)

    reconstruction_error = np.mean(np.square(X_all - X_all_reconstructed), axis=1)
    reconstruction_error_norm = robust_minmax_normalize(reconstruction_error)

    health_index = 1.0 - reconstruction_error_norm
    health_index = np.clip(health_index, 0.0, 1.0)

    df["reconstruction_error"] = reconstruction_error
    df["reconstruction_error_norm"] = reconstruction_error_norm
    df["health_index_ae"] = health_index

    return df, autoencoder, imputer, scaler


def add_health_label_from_hi(df: pd.DataFrame, hi_col: str = "health_index_ae") -> pd.DataFrame:
    """
    Crée un statut métier à partir du Health Index.
    """
    df = df.copy()

    if hi_col not in df.columns:
        raise ValueError(f"La colonne '{hi_col}' est absente.")

    conditions = [
        df[hi_col] >= 0.8,
        (df[hi_col] >= 0.5) & (df[hi_col] < 0.8),
        df[hi_col] < 0.5,
    ]
    labels = ["good", "warning", "critical"]

    df["health_status"] = np.select(conditions, labels, default="unknown")
    return df
