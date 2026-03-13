import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.predict import FEATURE_NAMES, FEATURE_LABELS

# ── Palettes dark / light ────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg":     "#0f172a",
        "card":   "#1e293b",
        "text":   "#f1f5f9",
        "muted":  "#94a3b8",
        "border": "#334155",
    },
    "light": {
        "bg":     "#f8fafc",
        "card":   "#ffffff",
        "text":   "#0f172a",
        "muted":  "#475569",
        "border": "#e2e8f0",
    },
}

ACCENT = "#3b82f6"
GREEN  = "#16a34a"
RED    = "#dc2626"
AMBER  = "#d97706"


def _layout(theme: str) -> dict:
    t = THEMES[theme]
    return dict(
        paper_bgcolor=t["bg"],
        plot_bgcolor=t["card"],
        font=dict(color=t["text"], family="sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


def plot_risk_gauge(proba: float, theme: str = "light") -> go.Figure:
    t = THEMES[theme]
    color = GREEN if proba < 0.30 else (AMBER if proba < 0.60 else RED)
    label = "FAIBLE" if proba < 0.30 else ("MODÉRÉ" if proba < 0.60 else "ÉLEVÉ")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": color},
            "bgcolor": t["card"],
            "bordercolor": t["border"],
            "steps": [
                {"range": [0,  30],  "color": "rgba(22, 163, 74, 0.15)"},
                {"range": [30, 60],  "color": "rgba(217, 119, 6, 0.15)"},
                {"range": [60, 100], "color": "rgba(220, 38, 38, 0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": proba * 100
            }
        },
        title={"text": f"<b>{label}</b>", "font": {"size": 28, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig.update_layout(**_layout(theme), height=250)
    return fig


def plot_feature_distributions(df: pd.DataFrame, user_values: dict, theme: str = "light") -> go.Figure:
    t = THEMES[theme]
    features = ["Glucose", "BMI", "Age", "BloodPressure"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[FEATURE_LABELS[f] for f in features])
    for i, feat in enumerate(features):
        row, col = divmod(i, 2); row += 1; col += 1
        fig.add_trace(go.Histogram(x=df[df["Outcome"]==0][feat], name="Non diabétique",
                                   marker_color=ACCENT, opacity=0.6, showlegend=(i==0), nbinsx=20), row=row, col=col)
        fig.add_trace(go.Histogram(x=df[df["Outcome"]==1][feat], name="Diabétique",
                                   marker_color=RED, opacity=0.6, showlegend=(i==0), nbinsx=20), row=row, col=col)
        if user_values.get(feat) is not None:
            fig.add_vline(x=user_values[feat], line_color=AMBER, line_width=2, line_dash="dash", row=row, col=col)
    fig.update_layout(**_layout(theme), height=480, barmode="overlay",
                      title_text="Votre profil comparé au dataset (ligne orange = vous)",
                      legend=dict(bgcolor=t["card"], bordercolor=t["border"], font=dict(color=t["text"])))
    fig.update_xaxes(showgrid=False, color=t["muted"])
    fig.update_yaxes(showgrid=False, color=t["muted"])
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    t = THEMES[theme]
    corr = df[FEATURE_NAMES + ["Outcome"]].corr()
    labels = [FEATURE_LABELS.get(f, f) for f in corr.columns]
    colorscale = ([[0, "#1e3a5f"], [0.5, t["card"]], [1, "#7f1d1d"]] if theme == "dark"
                  else [[0, "#bfdbfe"], [0.5, "#ffffff"], [1, "#fecaca"]])
    fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels,
                               colorscale=colorscale, zmin=-1, zmax=1,
                               text=corr.values.round(2), texttemplate="%{text}",
                               textfont={"size": 9, "color": t["text"]}))
    fig.update_layout(**_layout(theme), height=480, title_text="Corrélations entre les variables",
                      xaxis=dict(tickfont=dict(size=9, color=t["muted"])),
                      yaxis=dict(tickfont=dict(size=9, color=t["muted"])))
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, theme: str = "light") -> go.Figure:
    t = THEMES[theme]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"Régression Logistique (AUC = {auc:.3f})",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aléatoire",
                             line=dict(color=t["muted"], dash="dash", width=1)))
    fig.update_layout(**_layout(theme), height=400, title_text="Courbe ROC",
                      xaxis=dict(title="Taux de faux positifs", color=t["muted"]),
                      yaxis=dict(title="Taux de vrais positifs", color=t["muted"]))
    return fig


def plot_confusion_matrix(cm: np.ndarray, theme: str = "light") -> go.Figure:
    t = THEMES[theme]
    labels = ["Non diabétique (0)", "Diabétique (1)"]
    colorscale = ([[0, t["card"]], [1, ACCENT]] if theme == "dark"
                  else [[0, "#eff6ff"], [1, "#1d4ed8"]])
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=colorscale,
                               text=cm, texttemplate="%{text}",
                               textfont={"size": 18, "color": t["text"]}, showscale=False))
    fig.update_layout(**_layout(theme), height=350, title_text="Matrice de confusion",
                      xaxis=dict(title="Prédit", color=t["muted"]),
                      yaxis=dict(title="Réel", color=t["muted"]))
    return fig


def plot_class_distribution(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    t = THEMES[theme]
    counts = df["Outcome"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=["Non diabétique (0)", "Diabétique (1)"], y=counts.values,
                           marker_color=[GREEN, RED], text=counts.values,
                           textposition="outside", textfont=dict(color=t["text"])))
    fig.update_layout(**_layout(theme), height=350,
                      title_text="Distribution des classes dans le dataset",
                      yaxis=dict(color=t["muted"], showgrid=False), xaxis=dict(color=t["muted"]))
    return fig


def plot_shap_bar_themed(shap_vals: np.ndarray, theme: str = "light") -> plt.Figure:
    t = THEMES[theme]
    sv = np.array(shap_vals).flatten()
    labels = [FEATURE_LABELS[f] for f in FEATURE_NAMES]
    sorted_idx = np.argsort(np.abs(sv))
    sv_sorted = sv[sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]
    colors = [RED if v > 0 else GREEN for v in sv_sorted]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["card"])
    ax.barh(labels_sorted, sv_sorted, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Contribution SHAP", color=t["muted"], fontsize=10)
    ax.set_title("Impact de chaque variable sur votre résultat", color=t["text"], fontsize=12, pad=12)
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.spines[:].set_visible(False)
    ax.axvline(0, color=t["border"], linewidth=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(t["muted"])
    plt.tight_layout()
    return fig
