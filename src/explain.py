import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.predict import FEATURE_LABELS, FEATURE_NAMES

RECOMMENDATIONS = {
    "Glucose": {
        "high": "Votre taux de glucose est élevé. Réduisez la consommation de sucres raffinés et consultez un médecin pour un test d'hyperglycémie.",
        "low": "Votre taux de glucose est dans la normale. Continuez à maintenir une alimentation équilibrée."
    },
    "BMI": {
        "high": "Votre IMC suggère un surpoids. Même une perte de poids modérée (5-10%) peut réduire significativement le risque de diabète.",
        "low": "Votre IMC est dans la plage normale. Maintenez votre activité physique régulière."
    },
    "Age": {
        "high": "Le risque de diabète de type 2 augmente naturellement avec l'âge. Des bilans sanguins annuels sont recommandés.",
        "low": "À votre âge, adopter de bonnes habitudes de vie maintenant protège durablement votre santé."
    },
    "BloodPressure": {
        "high": "Une pression artérielle élevée est associée à un risque accru de diabète. Réduisez la consommation de sel et augmentez l'activité physique.",
        "low": "Votre pression artérielle est normale. Continuez à maintenir un mode de vie actif."
    },
    "Insulin": {
        "high": "Un taux d'insuline élevé peut signaler une résistance à l'insuline. Consultez votre médecin pour un bilan.",
        "low": "Votre taux d'insuline est dans la norme."
    },
    "DiabetesPedigreeFunction": {
        "high": "Vos antécédents familiaux représentent un facteur de risque. Des dépistages réguliers sont particulièrement importants pour vous.",
        "low": "Peu d'antécédents familiaux de diabète dans votre profil."
    }
}

def get_shap_explainer(model, X_train_scaled: np.ndarray):
    """Crée et retourne un explainer SHAP pour la régression logistique."""
    return shap.LinearExplainer(model, X_train_scaled, feature_names=FEATURE_NAMES)

def compute_shap_values(explainer, X_scaled: np.ndarray) -> np.ndarray:
    return explainer.shap_values(X_scaled)

def generate_natural_explanation(shap_vals: np.ndarray, input_df: pd.DataFrame, proba: float) -> str:
    sv = np.array(shap_vals).flatten()
    feature_names = FEATURE_NAMES

    sorted_idx = np.argsort(np.abs(sv))[::-1]
    top3 = sorted_idx[:3]

    risk_level = "faible" if proba < 0.30 else ("modéré" if proba < 0.60 else "élevé")

    lines = [f"<strong>Analyse de votre profil</strong> — Risque estimé : <strong>{proba:.0%}</strong> ({risk_level})"]
    lines.append("<br>Les 3 facteurs ayant le plus influencé ce résultat :")

    for rank, i in enumerate(top3, 1):
        fname = feature_names[i]
        label = FEATURE_LABELS[fname]
        direction = "augmente le risque" if sv[i] > 0 else "diminue le risque"
        contribution = round(abs(sv[i]) * 100, 1)
        #lines.append(f"<br><strong>{rank}. {label}</strong> — {direction} (contribution : {contribution}%)")
        lines.append(f"<br><strong>{rank}. {label}</strong> (valeur : {val:.1f}) — <em>{direction}</em> le risque (impact : {contribution:.3f})")

    return "\n".join(lines)

def generate_recommendations(shap_vals: np.ndarray, input_df: pd.DataFrame) -> list[dict]:
    """Génère des recommandations personnalisées basées sur les valeurs SHAP."""
    sv = np.array(shap_vals).flatten()
    recs = []

    thresholds = {
        "Glucose": 126,
        "BMI": 30,
        "Age": 45,
        "BloodPressure": 80,
        "Insulin": 200,
        "DiabetesPedigreeFunction": 0.5
    }

    for feat, threshold in thresholds.items():
        if feat not in FEATURE_NAMES:
            continue
        idx = FEATURE_NAMES.index(feat)
        val = input_df[feat].values[0]
        level = "high" if val >= threshold else "low"

        if feat in RECOMMENDATIONS:
            recs.append({
                "feature": FEATURE_LABELS[feat],
                "value": val,
                "level": level,
                "text": RECOMMENDATIONS[feat][level],
                "shap": sv[idx]
            })

    recs.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return recs[:4]

def plot_shap_bar(shap_vals: np.ndarray) -> plt.Figure:
    """Génère un graphique en barres des valeurs SHAP individuelles."""
    sv = np.array(shap_vals).flatten()
    labels = [FEATURE_LABELS[f] for f in FEATURE_NAMES]

    sorted_idx = np.argsort(np.abs(sv))
    sv_sorted = sv[sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in sv_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(labels_sorted, sv_sorted, color=colors, edgecolor="none", height=0.6)

    ax.set_xlabel("Contribution SHAP (impact sur la prédiction)", color="#94a3b8", fontsize=10)
    ax.set_title("Impact de chaque variable sur votre résultat", color="#f1f5f9", fontsize=12, pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.axvline(0, color="#475569", linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.label.set_color("#94a3b8")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#94a3b8")

    plt.tight_layout()
    return fig
