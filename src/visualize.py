import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.predict import FEATURE_NAMES, FEATURE_LABELS

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#f1f5f9"
MUTED = "#94a3b8"
ACCENT = "#3b82f6"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="monospace"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def plot_risk_gauge(proba: float) -> go.Figure:
    """Affiche une jauge circulaire de risque."""
    color = GREEN if proba < 0.30 else (AMBER if proba < 0.60 else RED)
    label = "FAIBLE" if proba < 0.30 else ("MODÉRÉ" if proba < 0.60 else "ÉLEVÉ")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": color},
            "bgcolor": CARD_BG,
            "bordercolor": CARD_BG,
            "steps": [
                {"range": [0, 30], "color": "rgba(20, 83, 45, 0.2)"},
                {"range": [30, 60], "color": "rgba(120, 53, 15, 0.2)"},
                {"range": [60, 100], "color": "rgba(127, 29, 29, 0.2)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": proba * 100
            }
        },
        title={"text": f"Niveau de risque : <b>{label}</b>", "font": {"size": 16, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]}
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
    )
    return fig

def plot_feature_distributions(df: pd.DataFrame, user_values: dict) -> go.Figure:
    """Compare le profil utilisateur aux distributions du dataset."""
    features = ["Glucose", "BMI", "Age", "BloodPressure"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[FEATURE_LABELS[f] for f in features])

    colors_0 = "#3b82f680"
    colors_1 = "#ef444480"

    for i, feat in enumerate(features):
        row, col = divmod(i, 2)
        row += 1
        col += 1

        non_diab = df[df["Outcome"] == 0][feat]
        diab = df[df["Outcome"] == 1][feat]

        fig.add_trace(go.Histogram(x=non_diab, name="Non diabétique",
                                   marker_color=colors_0, opacity=0.7,
                                   showlegend=(i == 0), nbinsx=20), row=row, col=col)
        fig.add_trace(go.Histogram(x=diab, name="Diabétique",
                                   marker_color=colors_1, opacity=0.7,
                                   showlegend=(i == 0), nbinsx=20), row=row, col=col)

        user_val = user_values.get(feat)
        if user_val is not None:
            fig.add_vline(x=user_val, line_color="#facc15", line_width=2,
                          line_dash="dash", row=row, col=col)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=480,
        barmode="overlay",
        title_text="Votre profil comparé au dataset (ligne jaune = vous)",
        legend=dict(bgcolor=CARD_BG, bordercolor=MUTED, font=dict(color=TEXT_COLOR))
    )
    fig.update_xaxes(showgrid=False, color=MUTED)
    fig.update_yaxes(showgrid=False, color=MUTED)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Matrice de corrélation des features."""
    corr = df[FEATURE_NAMES + ["Outcome"]].corr()
    labels = [FEATURE_LABELS.get(f, f) for f in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[[0, "#1e3a5f"], [0.5, CARD_BG], [1, "#7f1d1d"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 9, "color": TEXT_COLOR},
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=480,
                      title_text="Corrélations entre les variables",
                      xaxis=dict(tickfont=dict(size=9, color=MUTED)),
                      yaxis=dict(tickfont=dict(size=9, color=MUTED)))
    return fig

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """Courbe ROC du modèle."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"Régression Logistique (AUC = {auc:.3f})",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Aléatoire", line=dict(color=MUTED, dash="dash", width=1)))
    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                      title_text="Courbe ROC",
                      xaxis=dict(title="Taux de faux positifs", color=MUTED),
                      yaxis=dict(title="Taux de vrais positifs", color=MUTED))
    return fig

def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Matrice de confusion."""
    labels = ["Non diabétique (0)", "Diabétique (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0, CARD_BG], [1, ACCENT]],
        text=cm, texttemplate="%{text}",
        textfont={"size": 18, "color": TEXT_COLOR},
        showscale=False
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=350,
                      title_text="Matrice de confusion",
                      xaxis=dict(title="Prédit", color=MUTED),
                      yaxis=dict(title="Réel", color=MUTED))
    return fig

def plot_class_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribution des classes."""
    counts = df["Outcome"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=["Non diabétique (0)", "Diabétique (1)"],
        y=counts.values,
        marker_color=[GREEN, RED],
        text=counts.values,
        textposition="outside",
        textfont=dict(color=TEXT_COLOR)
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=350,
                      title_text="Distribution des classes dans le dataset",
                      yaxis=dict(color=MUTED, showgrid=False),
                      xaxis=dict(color=MUTED))
    return fig