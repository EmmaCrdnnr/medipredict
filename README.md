MediPredict — Prédiction du risque de diabète de type 2

Application web de sensibilisation développée dans le cadre du Mastère Data & IA — NEXA Digital School.

Mention légale
Cet outil est un **outil de sensibilisation uniquement**. Il ne constitue en aucun cas un diagnostic médical. Consultez un professionnel de santé pour tout avis médical.

Installation et lancement

```bash
# 1. Cloner le dépôt
git clone <votre-repo>
cd medipredict

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou : venv\Scripts\Activate.ps1  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer les fichiers nécessaires
# - model/medipredict_model.pkl  (depuis votre notebook)
# - model/scaler.pkl             (depuis votre notebook)
# - data/diabetes.csv            (dataset Pima Indians)

# 5. Lancer l'application
streamlit run app.py
```

Structure du projet

```
medipredict/
├── app.py                        # Point d'entrée Streamlit
├── model/
│   ├── medipredict_model.pkl     # Modèle entraîné (Régression Logistique)
│   └── scaler.pkl                # StandardScaler ajusté
├── src/
│   ├── predict.py                # Logique de prédiction
│   ├── explain.py                # Génération SHAP
│   └── visualize.py              # Fonctions de visualisation
├── data/
│   └── diabetes.csv              # Dataset Pima Indians
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

Déploiement sur Streamlit Cloud

1. Pushez votre code sur GitHub (dépôt public)
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Sélectionnez votre dépôt, branche `main`, fichier `app.py`
4. Cliquez sur **Deploy**

Conformité RGPD
- Aucune donnée utilisateur n'est stockée
- Consentement explicite requis avant utilisation
- Traitement en mémoire uniquement
- Mention légale visible sur la page d'accueil

Modèle
- **Algorithme** : Régression Logistique (scikit-learn)
- **Dataset** : Pima Indians Diabetes (768 rows, 8 features)
- **Explicabilité** : SHAP LinearExplainer
- **Choix éthique** : modèle interprétable préféré à Random Forest
