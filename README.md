# Micro-SIEM v2 — Classification du trafic reseau chiffre

Dashboard interactif pour l'analyse et la classification du trafic reseau chiffre. Utilise des modeles de Machine Learning pre-entraines pour detecter le trafic malveillant dans des sessions TCP/UDP.

## Fonctionnalites

| Page | Description |
|------|-------------|
| **Vue d'ensemble** | Import CSV/PCAP, classification automatique, cartes metriques, table des alertes, export PDF/CSV |
| **Analyse detaillee** | Explication SHAP par session, comparaison des features (Z-scores) |
| **Mode cascade** | Analyse multi-granularite : session → paquets pour les sessions incertaines |
| **Projection UMAP** | Visualisation 2D des clusters de sessions, faux negatifs, familles de malwares |
| **Configuration** | Seuil de detection configurable, toggle Isolation Forest, courbe FN/FP |
| **Statistiques** | Matrice de confusion, courbe ROC, feature importance, scores IF |
| **A propos** | Contexte du projet et description des modeles |

## Modeles

| Modele | Role | Performance |
|--------|------|-------------|
| **Random Forest (session)** | Classification principale (27 features) | F1 = 0.9950 |
| **Random Forest (paquets)** | Classification par paquet (21 features) | Accuracy = 99.98% |
| **XGBoost** | Explications SHAP (TreeExplainer) | F1 = 0.9898 |
| **Isolation Forest** | Detection d'anomalies non supervisee | Precision = 94.7% |

## Nouveautes V2

- **Mode cascade** : quand le RF session est incertain (P entre 0.3 et 0.7), descente au niveau paquet avec 3 strategies d'agregation (vote, proba moyenne, conservatif)
- **Ingestion PCAP** : import direct de captures reseau (.pcap), extraction automatique des features avec dpkt
- **Projection UMAP** : visualisation 2D interactive des clusters de sessions avec mise en evidence des faux negatifs
- **Export rapport** : PDF complet avec graphiques et metriques + CSV des resultats
- **Multi-dataset** : detection automatique du format CSV, support de fichiers non-CIC avec adaptation

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`.

## Utilisation

1. Cliquer sur **Charger donnees de demonstration** ou importer un CSV / PCAP
2. Naviguer entre les 7 pages via la sidebar
3. Explorer les explications SHAP dans **Analyse detaillee**
4. Tester le mode cascade pour affiner les sessions incertaines
5. Visualiser les clusters dans **Projection UMAP**
6. Ajuster le seuil dans **Configuration**
7. Exporter les resultats en PDF ou CSV

## Format d'entree

- **CSV** : 27 features session-based (CIC-Darknet2020). Colonne `label` optionnelle (0=benin, 1=malveillant)
- **PCAP** : fichier de capture reseau (.pcap). Les features sont extraites automatiquement
- **CSV paquets** : 21 features packet-based + colonne `unique_link_mark` (pour le mode cascade)

## Stack technique

Streamlit, scikit-learn, XGBoost, SHAP, Plotly, pandas, NumPy, Matplotlib, dpkt, UMAP, fpdf2

## Structure

```
micro-siem-encrypted-traffic/
├── .streamlit/
│   └── config.toml              # Theme et configuration Streamlit
├── app.py                        # Point d'entree (routing + sidebar)
├── src/
│   ├── models.py                 # Chargement des modeles ML
│   ├── ui_components.py          # CSS, cartes metriques, blocs d'explication
│   ├── cascade.py                # Logique du mode cascade
│   ├── projection.py             # UMAP / visualisation 2D
│   ├── report.py                 # Export PDF + CSV
│   └── feature_extraction.py     # Extraction PCAP + detection format
├── pages/
│   ├── overview.py               # Vue d'ensemble
│   ├── detail.py                 # Analyse detaillee (SHAP)
│   ├── cascade.py                # Mode cascade
│   ├── visualization.py          # Projection UMAP
│   ├── config.py                 # Configuration
│   ├── stats.py                  # Statistiques
│   └── about.py                  # A propos
├── models/
│   ├── model_random_forest.joblib    # RF session (37 Mo)
│   ├── model_rf_paquets.joblib       # RF paquets (31 Mo)
│   ├── model_xgboost.joblib          # XGBoost (355 Ko)
│   └── model_isolation_forest.joblib # IF (2 Mo)
├── data/
│   ├── feature_mapping.txt           # 27 features session
│   ├── packet_feature_mapping.txt    # 21 features paquets
│   ├── demo_sample.csv               # 5000 sessions demo
│   ├── demo_packets_sample.csv       # 45000 paquets demo
│   └── demo_umap_embedding.npz      # Projection UMAP pre-calculee
├── requirements.txt
└── README.md
```
