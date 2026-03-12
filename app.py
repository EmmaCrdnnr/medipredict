"""
MediPredict — Application Streamlit de prédiction du risque de diabète.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

#Imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import (
    FEATURE_NAMES, FEATURE_LABELS, FEATURE_RANGES,
    build_input_dataframe, predict, get_risk_level
)
from src.explain import (
    get_shap_explainer, compute_shap_values,
    generate_natural_explanation, generate_recommendations
)
from src.visualize import (
    plot_risk_gauge, plot_feature_distributions, plot_correlation_heatmap,
    plot_roc_curve, plot_confusion_matrix, plot_class_distribution,
    plot_shap_bar_themed
)

#Configuration de la page
st.set_page_config(
    page_title="MediPredict — Risque Diabète",
    layout="wide",
    initial_sidebar_state="expanded",
)

#CSS pour le style de la page (adaptatif light/dark)
st.markdown("""
<style>
.metric-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.legal-banner {
    border: 1px solid #3b82f6;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
}
.consent-box {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}
.risk-low  { color: #16a34a; font-weight: 700; }
.risk-mod  { color: #d97706; font-weight: 700; }
.risk-high { color: #dc2626; font-weight: 700; }
.rec-card {
    background: var(--secondary-background-color);
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.sidebar-label {
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    opacity: 0.5;
}
div[data-testid="stMetricValue"] { color: #3b82f6; }
.stButton>button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
}
.stButton>button:hover { background: linear-gradient(135deg, #2563eb, #3b82f6); transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

#Chargement du modèle
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("model/medipredict_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

@st.cache_data
def load_dataset():
    return pd.read_csv("data/diabetes.csv")

@st.cache_resource
def load_explainer(_model, _scaler, _df):
    """Prépare l'explainer SHAP avec les données d'entraînement."""
    from sklearn.model_selection import train_test_split
    X = _df.drop(columns=["Outcome"])
    y = _df["Outcome"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = _scaler.transform(X_train)
    return get_shap_explainer(_model, X_train_scaled), X_train_scaled

@st.cache_data
def compute_model_metrics(_model, _scaler, _df):
    """Calcule les métriques du modèle sur le jeu de test."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    X = _df.drop(columns=["Outcome"])
    y = _df["Outcome"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = _scaler.transform(X_test)
    y_pred = _model.predict(X_test_scaled)
    y_proba = _model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "fpr": fpr, "tpr": tpr, "cm": cm
    }


#Session
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

#Détection du thème
try:
    _base = st.get_option("theme.base")
    THEME = "dark" if _base == "dark" else "light"
except Exception:
    THEME = "light"



#Barre de navigation
with st.sidebar:
    st.markdown("## MediPredict")
    st.markdown('<p class="sidebar-label">NAVIGATION</p>', unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=["Accueil", "Mon profil de risque", "Comprendre ma prédiction", "Explorer les données"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<p class="sidebar-label">INFORMATIONS</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; line-height:1.6;">
    Modèle : <b style="color:#94a3b8">Régression Logistique</b><br>
    Dataset : <b style="color:#94a3b8">Pima Indians (768 obs.)</b><br>
    Version : <b style="color:#94a3b8">1.0.0</b>
    </div>
    """, unsafe_allow_html=True)


#Page 1
if page == "Accueil":
    st.markdown("# MediPredict")
    st.markdown("#### Outil de sensibilisation au risque de diabète de type 2")
    st.markdown("---")

    #Mention légale
    st.markdown("""
    <div class="legal-banner">
    <strong>Mention légale obligatoire</strong><br>
    Cet outil est un <strong>outil de sensibilisation uniquement</strong>. Il ne constitue
    en aucun cas un avis médical, un diagnostic ou une prescription. Les résultats fournis
    sont des estimations statistiques basées sur un dataset de recherche.
    <strong>En cas de doute ou d'inquiétude concernant votre santé, consultez impérativement
    un professionnel de santé qualifié.</strong>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Objectif</h4>
        <p style="color:#94a3b8; font-size:0.9rem;">
        Évaluer votre profil de risque de diabète de type 2 à partir
        d'indicateurs de santé anonymes et recevoir des explications claires.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Vos données</h4>
        <p style="color:#94a3b8; font-size:0.9rem;">
        Aucune donnée personnelle n'est stockée ni transmise. Tout le
        traitement s'effectue en mémoire, localement sur le serveur.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>Notre modèle</h4>
        <p style="color:#94a3b8; font-size:0.9rem;">
        Régression Logistique entraînée sur le dataset Pima Indians.
        Choisi pour son interprétabilité et sa conformité éthique.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###Politique de confidentialité")
    st.markdown("""
    <div style="background:#1e293b; border-radius:8px; padding:1rem 1.5rem; color:#94a3b8; font-size:0.88rem; line-height:1.8;">
    • Les données saisies dans ce formulaire sont utilisées <strong>uniquement</strong> pour calculer votre estimation de risque.<br>
    • Ces données ne sont <strong>ni enregistrées, ni transmises à des tiers, ni utilisées à des fins commerciales</strong>.<br>
    • Le traitement est intégralement réalisé en mémoire vive (RAM) et effacé à la fermeture de la session.<br>
    • Conformément au RGPD (Art. 5), vous bénéficiez d'un droit d'accès, de rectification et d'effacement de vos données.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###Consentement")
    st.markdown("""
    <div class="consent-box">
    Avant d'utiliser MediPredict, vous devez confirmer avoir lu et compris :
    </div>
    """, unsafe_allow_html=True)

    consent = st.checkbox(
        "J'ai lu la mention légale et la politique de confidentialité. Je comprends que cet outil "
        "est uniquement un outil de sensibilisation, non un outil médical. Je consens au traitement "
        "temporaire de mes données de santé anonymes pour calculer mon estimation de risque."
    )

    if consent:
        st.session_state.consent_given = True
        st.success("Consentement enregistré. Vous pouvez accéder à l'analyse via le menu.")
    else:
        st.session_state.consent_given = False
        st.info("Veuillez cocher la case ci-dessus pour accéder aux fonctionnalités.")


#Page 2
elif page == "Mon profil de risque":
    st.markdown("#Mon profil de risque")
    st.markdown("Renseignez vos indicateurs de santé pour obtenir une estimation personnalisée.")

    if not st.session_state.consent_given:
        st.warning("Veuillez d'abord donner votre consentement sur la page **Accueil**.")
        st.stop()

    #Chargement modèle
    try:
        model, scaler = load_model_and_scaler()
    except FileNotFoundError:
        st.error("Modèle non trouvé. Assurez-vous que `model/medipredict_model.pkl` et `model/scaler.pkl` sont présents.")
        st.stop()

    st.markdown("""
    <div class="legal-banner">
    ℹRemplissez les champs ci-dessous avec vos valeurs médicales. Si une valeur ne s'applique pas,
    laissez la valeur par défaut ou entrez 0. Ces données ne seront pas conservées.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 1])

    with col_form:
        with st.form("profil_form"):
            st.markdown("####Vos informations de santé")

            #Gestion des pregnancies
            st.markdown("**Nombre de grossesses**")
            preg_na = st.checkbox("Non applicable (homme ou sans grossesse)", key="preg_na")
            if preg_na:
                pregnancies = 0
                st.info("Valeur fixée à 0.")
            else:
                pregnancies = st.number_input(
                    "Nombre de grossesses",
                    min_value=0, max_value=17, value=3, step=1,
                    help="Nombre de fois que vous avez été enceinte",
                    label_visibility="collapsed"
                )

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                glucose = st.number_input(
                    "Taux de glucose (mg/dL)",
                    min_value=44, max_value=199, value=120, step=1,
                    help="Concentration en glucose plasmatique 2h après un test de tolérance au glucose (44–199 mg/dL)"
                )
                bp = st.number_input(
                    "Pression artérielle diastolique (mm Hg)",
                    min_value=24, max_value=122, value=72, step=1,
                    help="Pression artérielle diastolique mesurée en mm Hg (24–122)"
                )
                skin = st.number_input(
                    "Épaisseur du pli cutané tricipital (mm)",
                    min_value=7, max_value=99, value=29, step=1,
                    help="Épaisseur du pli cutané du triceps en mm (7–99)"
                )
                insulin = st.number_input(
                    "Insuline sérique 2h (µU/mL)",
                    min_value=14, max_value=846, value=80, step=1,
                    help="Concentration en insuline sérique 2h après test (14–846 µU/mL)"
                )

            with col2:
                bmi = st.number_input(
                    "Indice de masse corporelle — IMC (kg/m²)",
                    min_value=18.2, max_value=67.1, value=32.0, step=0.1,
                    help="IMC = Poids(kg) / Taille²(m). Normal : 18.5–24.9 ; Surpoids : 25–29.9"
                )
                dpf = st.number_input(
                    "Score d'antécédents familiaux (DiabetesPedigreeFunction)",
                    min_value=0.078, max_value=2.42, value=0.47, step=0.001,
                    help="Score calculé à partir des antécédents familiaux de diabète (0.078–2.42)"
                )
                age = st.number_input(
                    "Âge (années)",
                    min_value=21, max_value=81, value=33, step=1,
                    help="Votre âge en années"
                )

            submitted = st.form_submit_button("Analyser mon profil", type="primary", use_container_width=True)

    with col_info:
        st.markdown("####Guide de saisie")
        st.markdown("""
        <div style="background:#1e293b; border-radius:8px; padding:1rem; font-size:0.82rem; color:#94a3b8; line-height:1.9;">
        <b style="color:#f1f5f9">IMC normal</b><br>18.5 – 24.9<br><br>
        <b style="color:#f1f5f9">Glucose à jeun</b><br>Normal &lt; 100 mg/dL<br>Prédiabète : 100–125<br>Diabète ≥ 126<br><br>
        <b style="color:#f1f5f9">Pression diastolique</b><br>Normale &lt; 80 mm Hg<br><br>
        <b style="color:#f1f5f9">Score familial</b><br>Faible : &lt; 0.3<br>Moyen : 0.3–0.7<br>Élevé : &gt; 0.7
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        user_values = {
            "Pregnancies": float(pregnancies),
            "Glucose": float(glucose),
            "BloodPressure": float(bp),
            "SkinThickness": float(skin),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": float(age),
        }

        #Validation côté serveur
        valid = True
        for feat, val in user_values.items():
            mn, mx, _, _ = FEATURE_RANGES[feat]
            if val < mn or val > mx:
                st.error(f"Valeur hors plage pour **{FEATURE_LABELS[feat]}** : {val} (plage attendue : {mn}–{mx})")
                valid = False

        if valid:
            input_df = build_input_dataframe(user_values)
            proba, classe = predict(model, scaler, input_df)
            risk_level, risk_color, risk_desc = get_risk_level(proba)

            st.session_state.prediction_done = True
            st.session_state.last_result = {
                "user_values": user_values,
                "input_df": input_df,
                "proba": proba,
                "classe": classe,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "risk_desc": risk_desc,
            }

    #Affichage du résultat
    if st.session_state.prediction_done and st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown("---")
        st.markdown("##Votre résultat")

        col_gauge, col_text = st.columns([1, 2])
        with col_gauge:
            fig_gauge = plot_risk_gauge(r["proba"], theme=THEME)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_text:
            st.markdown(f"""
            <div class="metric-card">
            <h3 style="color:{r['risk_color']}">Risque {r['risk_level']}</h3>
            <p style="color:#94a3b8; font-size:1rem;">{r['risk_desc']}</p>
            <br>
            <p style="color:#64748b; font-size:0.82rem;">
            Ce résultat est une <strong>estimation statistique</strong>, non un diagnostic médical.
            La probabilité estimée reflète la similitude de votre profil avec les cas du dataset
            d'entraînement (population amérindienne Pima). Consultez un médecin pour un bilan personnalisé.
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.info("Rendez-vous sur la page **Comprendre ma prédiction** pour les détails et recommandations.")


#Page 3
elif page == "Comprendre ma prédiction":
    st.markdown("#Comprendre ma prédiction")

    if not st.session_state.consent_given:
        st.warning("Veuillez d'abord donner votre consentement sur la page **Accueil**.")
        st.stop()

    if not st.session_state.prediction_done or not st.session_state.last_result:
        st.info("Commencez par remplir votre profil sur la page **Mon profil de risque**.")
        st.stop()

    try:
        model, scaler = load_model_and_scaler()
        df = load_dataset()
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e}")
        st.stop()

    r = st.session_state.last_result
    input_df = r["input_df"]

    # Calcul SHAP
    with st.spinner("Calcul des explications SHAP en cours..."):
        explainer, X_train_scaled = load_explainer(model, scaler, df)
        X_user_scaled = scaler.transform(input_df)
        shap_vals = compute_shap_values(explainer, X_user_scaled)

    # Explication naturelle
    st.markdown("###Explication de votre résultat")
    explanation = generate_natural_explanation(shap_vals[0], input_df, r["proba"])
    st.markdown(f"""
    <div class="metric-card">
    {explanation.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Graphique SHAP
    st.markdown("###Impact de chaque variable (SHAP)")
    st.markdown('<p style="color:#64748b; font-size:0.85rem;">Rouge = augmente le risque &nbsp;|&nbsp;Vert = diminue le risque</p>', unsafe_allow_html=True)
    fig_shap = plot_shap_bar(shap_vals[0], theme=THEME)
    st.pyplot(fig_shap, use_container_width=True)

    st.markdown("---")

    # Comparaison avec le dataset
    st.markdown("###Votre profil comparé au dataset")
    st.markdown('<p style="color:#64748b; font-size:0.85rem;">La ligne jaune pointillée représente votre valeur.</p>', unsafe_allow_html=True)
    fig_dist = plot_feature_distributions(df, r["user_values"], theme=THEME)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    #Recommandations
    st.markdown("###Recommandations personnalisées")
    recs = generate_recommendations(shap_vals[0], input_df)
    for rec in recs:
        level_icon = "🔴" if rec["level"] == "high" else "🟢"
        st.markdown(f"""
        <div class="rec-card">
        {level_icon} <strong>{rec['feature']}</strong> (valeur : {rec['value']:.1f})<br>
        {rec['text']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem; padding:1rem; background:#1e293b; border-radius:8px; font-size:0.82rem; color:#64748b;">
    Ces recommandations sont génériques et basées sur des critères épidémiologiques généraux.
    Elles ne remplacent pas un avis médical personnalisé. Consultez un professionnel de santé
    pour toute décision médicale.
    </div>
    """, unsafe_allow_html=True)


#Page 4
elif page == "Explorer les données":
    st.markdown("#Explorer les données")
    st.markdown("Visualisez le dataset et les performances du modèle en toute transparence.")

    try:
        model, scaler = load_model_and_scaler()
        df = load_dataset()
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Statistiques du dataset", "Performance du modèle", "Transparence & Biais"])

    with tab1:
        st.markdown("#### Distribution des classes")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Observations totales", len(df))
            st.metric("Non diabétiques", int((df["Outcome"] == 0).sum()))
            st.metric("Diabétiques", int((df["Outcome"] == 1).sum()))
            st.metric("Features", len(FEATURE_NAMES))
        with col2:
            fig_dist = plot_class_distribution(df, theme=THEME)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Statistiques descriptives")
        desc = df[FEATURE_NAMES].describe().T.round(2)
        st.dataframe(desc, use_container_width=True)

        st.markdown("#### Matrice de corrélations")
        fig_corr = plot_correlation_heatmap(df, theme=THEME)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        metrics = compute_model_metrics(model, scaler, df)

        st.markdown("#### Métriques sur le jeu de test (20%)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Précision", f"{metrics['precision']:.3f}")
        c3.metric("Rappel", f"{metrics['recall']:.3f}")
        c4.metric("F1-Score", f"{metrics['f1']:.3f}")
        c5.metric("AUC-ROC", f"{metrics['auc']:.3f}")

        col_roc, col_cm = st.columns(2)
        with col_roc:
            fig_roc = plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc"], theme=THEME)
            st.plotly_chart(fig_roc, use_container_width=True)
        with col_cm:
            fig_cm = plot_confusion_matrix(metrics["cm"], theme=THEME)
            st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.markdown("####Description du modèle")
        st.markdown("""
        <div class="metric-card">
        <p style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
        <strong style="color:#f1f5f9">Algorithme :</strong> Régression Logistique (scikit-learn, max_iter=1000)<br>
        <strong style="color:#f1f5f9">Choix éthique :</strong> Préféré à Random Forest pour son interprétabilité native
        (coefficients directement lisibles) et sa stabilité sur un dataset de taille limitée (768 obs.).<br>
        <strong style="color:#f1f5f9">Normalisation :</strong> StandardScaler (moyenne=0, écart-type=1)<br>
        <strong style="color:#f1f5f9">Reproductibilité :</strong> random_state=42 fixé sur toutes les opérations aléatoires.<br>
        <strong style="color:#f1f5f9">Explicabilité :</strong> SHAP LinearExplainer — valeurs exactes (non approximées).
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("####Limites et biais identifiés")
        st.markdown("""
        <div style="background:#1e293b; border-radius:8px; padding:1.2rem 1.5rem; color:#94a3b8; font-size:0.88rem; line-height:2;">
        <span style="color:#ef4444">●</span> <strong style="color:#f1f5f9">Biais de représentation :</strong>
        Le dataset Pima Indians est composé exclusivement de femmes amérindiennes de plus de 21 ans.
        Il n'est pas représentatif de la population française générale (hommes, autres ethnies, autres tranches d'âge).<br>
        <span style="color:#f59e0b">●</span> <strong style="color:#f1f5f9">Biais d'âge :</strong>
        Les performances du modèle sont légèrement inférieures sur les moins de 30 ans (sous-représentés dans le dataset).<br>
        <span style="color:#f59e0b">●</span> <strong style="color:#f1f5f9">Variable Pregnancies :</strong>
        Non applicable aux hommes et aux femmes sans grossesses. Une option "Non applicable" est proposée dans le formulaire.<br>
        <span style="color:#22c55e">●</span> <strong style="color:#f1f5f9">Mesure corrective :</strong>
        Ces limites sont clairement communiquées à l'utilisateur. L'outil est présenté comme un outil de sensibilisation,
        non comme un diagnostic.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("####Conformité RGPD")
        st.markdown("""
        <div style="background:#1e293b; border-radius:8px; padding:1.2rem 1.5rem; color:#94a3b8; font-size:0.88rem; line-height:2;">
        <span style="color:#22c55e">✓</span> Consentement explicite recueilli avant tout traitement<br>
        <span style="color:#22c55e">✓</span> Aucune donnée personnelle stockée (traitement en mémoire uniquement)<br>
        <span style="color:#22c55e">✓</span> Finalité clairement définie (sensibilisation uniquement)<br>
        <span style="color:#22c55e">✓</span> Mention légale visible sur la page d'accueil<br>
        <span style="color:#22c55e">✓</span> Base légale : consentement (Art. 6.1.a RGPD)<br>
        <span style="color:#22c55e">✓</span> Validation des entrées côté serveur (plages biologiquement plausibles)
        </div>
        """, unsafe_allow_html=True)
