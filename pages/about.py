"""
Page 7 : A propos — Contexte, modeles, pipeline.
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
    - **Features** : 27 caracteristiques de session selectionnees par importance (Gini + Cohen's d)
    - **Labels** : 0 = trafic benin, 1 = trafic malveillant (Tor, VPN malveillant)

    ---

    ### Pipeline d'analyse complet

    Ce dashboard est l'aboutissement d'un projet en 5 axes :
    1. **Analyse packet-based** — Classification par paquets individuels (99.98% accuracy)
    2. **Comparaison d'algorithmes** — Random Forest vs XGBoost vs MLP
    3. **Interpretabilite** — SHAP : comprendre pourquoi le modele decide
    4. **Visualisation** — t-SNE, UMAP, clustering K-Means des familles de malwares
    5. **Detection d'anomalies** — Isolation Forest non supervise

    ### Fonctionnalites V2

    - **Mode cascade** — Quand le RF session est incertain, descente au niveau paquet pour affiner
    - **Ingestion PCAP** — Import direct de captures reseau, extraction automatique des features
    - **Projection UMAP** — Visualisation 2D des clusters de sessions avec faux negatifs
    - **Export rapport** — PDF complet avec graphiques + CSV des resultats
    - **Multi-dataset** — Support de formats CSV varies avec detection automatique

    ### Technologies

    Streamlit, scikit-learn, XGBoost, SHAP, Plotly, pandas, NumPy, Matplotlib, dpkt, UMAP, fpdf2
    """)

    st.markdown("---")
    st.caption("Projet realise par Loris Dietrich — Analyse de cybersecurite")
