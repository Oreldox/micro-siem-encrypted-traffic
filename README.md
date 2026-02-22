# Micro-SIEM — Classification du trafic reseau chiffre

Dashboard interactif pour l'analyse et la classification du trafic reseau chiffre. Utilise des modeles de Machine Learning pre-entraines pour detecter le trafic malveillant dans des sessions TCP/UDP.

## Fonctionnalites

| Page | Description |
|------|-------------|
| **Vue d'ensemble** | Import CSV, classification automatique, cartes metriques, table des alertes |
| **Analyse detaillee** | Explication SHAP par session, comparaison des features (Z-scores) |
| **Configuration** | Seuil de detection configurable, toggle Isolation Forest, courbe FN/FP |
| **Statistiques** | Matrice de confusion, courbe ROC, feature importance, scores IF |
| **A propos** | Contexte du projet et description des modeles |

## Modeles

| Modele | Role | Performance |
|--------|------|-------------|
| **Random Forest** | Classification principale (session-based) | F1 = 0.9950 |
| **XGBoost** | Explications SHAP (TreeExplainer) | F1 = 0.9898 |
| **Isolation Forest** | Detection d'anomalies non supervisee | Precision = 94.7% |

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`.

## Utilisation

1. Cliquer sur **Charger donnees de demonstration** ou importer un CSV
2. Naviguer entre les pages via la sidebar
3. Ajuster le seuil de detection et activer/desactiver l'Isolation Forest dans **Configuration**
4. Explorer les explications SHAP dans **Analyse detaillee**

## Format d'entree

CSV avec les 27 features session-based. Si une colonne `label` est presente (0=benin, 1=malveillant), le dashboard affiche les metriques de performance.

## Stack technique

Streamlit, scikit-learn, XGBoost, SHAP, Plotly, pandas, NumPy, Matplotlib

## Structure

```
micro-siem-encrypted-traffic/
├── .streamlit/
│   └── config.toml          # Theme et configuration Streamlit
├── models/
│   ├── model_random_forest.joblib
│   ├── model_xgboost.joblib
│   └── model_isolation_forest.joblib
├── data/
│   ├── feature_mapping.txt
│   ├── packet_feature_mapping.txt
│   └── demo_sample.csv      # 5000 sessions pour la demonstration
├── app.py                    # Application principale
├── requirements.txt
└── README.md
```
