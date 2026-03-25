import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée quelques variables dérivées simples.
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

    return df


def build_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la cible continue health_score entre 0 et 1.
    1 = machine en excellente santé
    0 = machine critique
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if not {"machine_id", "timestamp"}.issubset(df.columns):
        raise ValueError(
            "Les colonnes 'machine_id' et 'timestamp' sont nécessaires pour construire le health_score."
        )

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df["vibration_delta"] = df.groupby("machine_id")["vibration_rms"].diff()
    df["temperature_delta"] = df.groupby("machine_id")["temperature_motor"].diff()

    df["vibration_delta"] = df["vibration_delta"].fillna(0)
    df["temperature_delta"] = df["temperature_delta"].fillna(0)

    df["anomaly_trend_raw"] = (
        0.6 * df["vibration_delta"].clip(lower=0)
        + 0.4 * df["temperature_delta"].clip(lower=0)
    )

    score_cols = [
        "vibration_rms",
        "temperature_motor",
        "pressure_level",
        "rpm",
        "anomaly_trend_raw",
    ]

    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[score_cols])

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[f"{col}_norm" for col in score_cols],
        index=df.index,
    )

    df = pd.concat([df, scaled_df], axis=1)

    df["health_score"] = 1 - (
        0.30 * df["vibration_rms_norm"]
        + 0.25 * df["temperature_motor_norm"]
        + 0.20 * df["pressure_level_norm"]
        + 0.15 * df["rpm_norm"]
        + 0.10 * df["anomaly_trend_raw_norm"]
    )

    df["health_score"] = df["health_score"].clip(0, 1)

    return df


def add_health_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un statut métier à partir du health_score.
    """
    df = df.copy()

    if "health_score" not in df.columns:
        raise ValueError("La colonne 'health_score' est absente.")

    conditions = [
        df["health_score"] >= 0.8,
        (df["health_score"] >= 0.5) & (df["health_score"] < 0.8),
        df["health_score"] < 0.5,
    ]
    labels = ["good", "warning", "critical"]

    df["health_status"] = np.select(conditions, labels, default="unknown")

    return df