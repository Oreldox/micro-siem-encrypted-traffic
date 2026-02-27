"""
Page : A propos — Contexte, modeles, pipeline, historique des versions.
"""

import streamlit as st


def render():
    st.header("A propos de cet outil")

    st.markdown("""
    ### Qu'est-ce que c'est ?

    Ce dashboard est un **Micro-SIEM** (Security Information and Event Management) specialise
    dans la detection de trafic reseau malveillant **chiffre**. Il analyse les metadonnees des
    sessions reseau (duree, volume, timing, nombre de paquets) sans avoir besoin de dechiffrer
    le contenu.

    ### Pourquoi c'est utile ?

    Avec plus de 90% du trafic web chiffre en TLS/SSL, les outils classiques d'inspection
    de contenu (DPI) sont aveugles. Ce dashboard utilise le **Machine Learning** pour
    detecter des comportements suspects (C2, exfiltration, ransomware) a partir des
    caracteristiques statistiques des connexions.

    ---

    ### Modeles utilises

    | Modele | Role | Performance |
    |--------|------|-------------|
    | **Random Forest** | Classification principale (27 features) | **F1 = 0.9950** (99.5% accuracy) |
    | **XGBoost** | Explications SHAP par session | F1 = 0.9898 |
    | **Isolation Forest** | Detection d'anomalies non supervisee | Precision = 94.7% |
    | **Random Forest (paquets)** | Classification par paquet (21 features) | Accuracy = 99.98% |

    ### Donnees d'entrainement

    - **Dataset** : CIC-Darknet2020 (244 000 sessions d'entrainement, 122 000 de test)
    - **Classes** : 304 327 sessions benignes (4 datasets) + 306 329 sessions malveillantes (25 familles)
    - **Features** : 27 caracteristiques de session selectionnees par Cohen's d + correlation de Pearson
    - **Labels** : 0 = trafic benin, 1 = trafic malveillant (Tor, VPN malveillant)

    ---

    ### Pipeline d'analyse complet

    Ce dashboard est l'aboutissement d'un projet d'analyse en **7 axes** :

    1. **Selection des features** — Pipeline Cohen's d (>= 0.45) + Pearson (<= 0.85) : 280 colonnes -> 27 features
    2. **Analyse packet-based** — Classification par paquets individuels (RF 99.98%, feature dominante : Time_cost 61.77%)
    3. **Comparaison d'algorithmes** — Random Forest vs XGBoost vs MLP (RF gagne avec F1=0.9950)
    4. **Interpretabilite SHAP** — TreeExplainer sur XGBoost, beeswarm, dependence plots, 8/10 features communes avec Gini
    5. **Analyse des faux negatifs** — 511 FN (0.84%), proba moyenne 0.311, 89.6% concentres dans un cluster K-Means
    6. **Clustering** — K-Means (K=2, silhouette=0.4718) + DBSCAN (22 clusters, 3.6% bruit)
    7. **Detection d'anomalies** — Isolation Forest non supervise, complementarite RF + IF analysee

    ---

    ### Performances de reference

    | Metrique | Random Forest | XGBoost | MLP |
    |----------|---------------|---------|-----|
    | Accuracy | **99.50%** | 98.93% | 99.14% |
    | Precision | 99.84% | 98.62% | 99.87% |
    | Recall | 99.16% | 99.24% | 98.41% |
    | F1-score | **0.9950** | 0.9893 | 0.9914 |
    | FN | 511 | 465 | 969 |
    | FP | 98 | 847 | 77 |

    - **AUC** : 0.999857 (ROC)
    - **Youden's J** : seuil optimal = 0.3959 (TPR = 0.9959, FPR = 0.0000)

    ---

    ### Isolation Forest — Complementarite avec RF

    L'IF est entraine uniquement sur le trafic benin. Resultat :
    - RF exclusif (sans IF) : **48 914** malwares detectes
    - IF exclusif (sans RF) : **1** malware detecte
    - Chevauchement RF et IF : **11 434** malwares detectes par les deux
    - Manques par les deux : **510** (FN du systeme combine)

    Verdict : l'IF ne recupere que 1 FN supplementaire au prix de +637 FP.
    Le RF domine largement sur la detection des malwares connus.

    ---
    """)

    st.markdown("""
    ### Historique des versions

    #### V1 — Prototype (2024)
    - Dashboard Streamlit basique
    - Classification RF session (27 features)
    - Import CSV CIC-Darknet2020 uniquement

    #### V2 — Multi-format + Cascade
    - Mode cascade : quand le RF session est incertain, descente au niveau paquet pour affiner
    - Ingestion PCAP/PCAPNG direct avec extraction automatique des features
    - Projection UMAP — Visualisation 2D des clusters de sessions
    - Export rapport PDF + CSV
    - Support multi-format (CSV, Excel)

    #### V3 — Confiance + Accord
    - SHAP sur le Random Forest (F1=0.995) au lieu de XGBoost
    - Score de confiance par prediction (marge + qualite features + accord modeles)
    - Accord inter-modeles RF vs XGBoost avec detection des desaccords
    - Analyse temporelle (beaconing C2, bursts, connexions persistantes)
    - Calibration de seuil adaptee a la qualite des donnees importees
    - Z-scores vs statistiques d'entrainement
    - Feedback utilisateur (faux positifs/negatifs manuels)

    #### V4 — Formats etendus
    - Support PCAPNG (detection magic bytes)
    - Import CSV Wireshark (Export Packet Dissections) avec aggregation paquet -> session
    - Import CSV tshark (champs etendus ip.ttl, tcp.window_size, etc.)
    - Import CICFlowMeter (mapping 80+ colonnes)
    - Import Excel (.xlsx, .xls) et Zeek conn.log
    - Detection de format intelligente (paquet-level vs session-level)
    - Indicateur de qualite (X/27 features) avec code couleur
    - Optimisation du seuil (courbe F1/Precision/Recall)

    #### V5 — Analyse complete + Test externe (actuelle)
    - **Test sur dataset externe** : tester le modele sur des donnees jamais vues, avec comparaison aux performances d'entrainement
    - **Page Methodologie** : pipeline de selection des features (Cohen's d + Pearson), visualisation interactive
    - **Page SHAP Global** : beeswarm, dependence plots, comparaison SHAP vs Gini, comparaison RF/XGBoost/MLP
    - **Page Faux negatifs** : analyse des 511 FN, profils comparatifs, projection t-SNE
    - **Page Clustering** : K-Means + DBSCAN, lien avec les faux negatifs, clustering dynamique
    - **Youden's J** : seuil optimal identifie sur la courbe ROC
    - **12 pages** au total avec navigation structuree
    """)

    st.markdown("""
    ---

    ### Formats supportes

    | Format | Type | Features | Qualite |
    |--------|------|----------|---------|
    | **PCAP / PCAPNG** | Capture brute | 27/27 | Optimal |
    | **CSV CIC-Darknet2020** | Session-level | 27/27 | Optimal |
    | **CSV CICFlowMeter** | Session-level | ~20/27 | Bon |
    | **CSV Wireshark** | Paquet-level | 10-27/27 | Variable |
    | **CSV tshark (etendu)** | Paquet-level | 27/27 | Optimal |
    | **Excel (.xlsx)** | Auto-detecte | Variable | Variable |
    | **Zeek conn.log** | Session-level | ~5/27 | Partiel |

    ### Technologies

    Streamlit, scikit-learn, XGBoost, SHAP, Plotly, pandas, NumPy, Matplotlib, dpkt, UMAP, fpdf2, openpyxl
    """)

    st.markdown("---")
    st.caption("Projet realise par Loris Dietrich — Analyse de cybersecurite")
