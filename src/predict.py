import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

FEATURE_LABELS = {
    "Pregnancies": "Nombre de grossesses",
    "Glucose": "Taux de glucose",
    "BloodPressure": "Pression artérielle",
    "SkinThickness": "Épaisseur dde la peau",
    "Insulin": "Taux d'insuline",
    "BMI": "Indice de masse corporelle",
    "DiabetesPedigreeFunction": "Antécédents familiaux",
    "Age": "Âge"
}

FEATURE_RANGES = {
    "Pregnancies": (0, 17, 3, "Entrez 0 si non applicable"),
    "Glucose": (44, 199, 120, "Taux de glucose à jeun"),
    "BloodPressure": (24, 122, 72, "Pression diastolique"),
    "SkinThickness": (7, 99, 29, "Épaisseur de la peau"),
    "Insulin": (14, 846, 80, "Taux d'insuline"),
    "BMI": (18.2, 67.1, 32.0, "Poids (kg) / Taille² (m)"),
    "DiabetesPedigreeFunction": (0.078, 2.42, 0.47, "Score basé sur les antécédents familiaux"),
    "Age": (21, 81, 33, "Votre âge en années")
}

def get_risk_level(proba: float) -> tuple[str, str, str]:
    """Retourne (niveau, couleur, description) selon la probabilité."""
    if proba < 0.30:
        return "Faible", "#22c55e", "Votre profil présente un faible risque de diabète de type 2."
    elif proba < 0.60:
        return "Modéré", "#f59e0b", "Votre profil présente un risque modéré. Une surveillance est conseillée."
    else:
        return "Élevé", "#ef4444", "Votre profil présente un risque élevé. Consultez un professionnel de santé."

def build_input_dataframe(values: dict) -> pd.DataFrame:
    """Construit un DataFrame à partir d'un dictionnaire de valeurs."""
    row = {feat: [values.get(feat, 0)] for feat in FEATURE_NAMES}
    return pd.DataFrame(row)

def predict(model, scaler, input_df: pd.DataFrame) -> tuple[float, int]:
    """Retourne (probabilité, classe prédite)."""
    X_scaled = scaler.transform(input_df)
    proba = model.predict_proba(X_scaled)[0, 1]
    classe = int(proba >= 0.5)
    return float(proba), classe
