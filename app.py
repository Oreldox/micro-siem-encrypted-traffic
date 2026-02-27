"""
Micro-SIEM Dashboard V5 — Classification du trafic reseau chiffre.
Point d'entree principal : routing, sidebar, configuration globale.
"""

import streamlit as st
import numpy as np

# === PAGE CONFIG (doit etre le premier appel Streamlit) ===
st.set_page_config(
    page_title="Micro-SIEM | Trafic Chiffre",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ui_components import inject_css
from src.models import (
    load_models, load_feature_mapping,
    SESSION_MAPPING_PATH, PACKET_MAPPING_PATH
)

# === CSS ===
inject_css()

# === CHARGEMENT MODELES ===
models, model_info = load_models()
session_features = load_feature_mapping(SESSION_MAPPING_PATH)
packet_features = load_feature_mapping(PACKET_MAPPING_PATH)

# === SIDEBAR ===
st.sidebar.markdown('<div class="sidebar-title">\U0001f6e1\ufe0f Micro-SIEM</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Classification du trafic chiffre</div>',
                    unsafe_allow_html=True)

st.sidebar.markdown("**Analyse**")
page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Test externe",
    "Analyse detaillee",
    "Mode cascade",
    "Projection",
    "SHAP Global",
    "Faux negatifs",
    "Clustering",
    "Methodologie",
    "Configuration",
    "Statistiques",
    "A propos"
], label_visibility="collapsed")

st.sidebar.divider()

# Config rapide
st.sidebar.markdown("**Parametres**")
threshold = st.sidebar.slider("Seuil de detection", 0.0, 1.0, 0.5, 0.01,
                              key="sidebar_threshold",
                              help="Sessions avec P(malveillant) >= seuil = alerte")
use_if = st.sidebar.toggle("Isolation Forest", False, key="sidebar_if",
                           help="Detection d'anomalies non supervisee")

config = {"threshold": threshold, "use_if": use_if}

# === COMPTEUR D'ALERTES EN TEMPS REEL ===
if "probas" in st.session_state:
    st.sidebar.divider()
    probas_live = st.session_state["probas"]
    n_alerts_rf = int((probas_live >= threshold).sum())

    if use_if and st.session_state.get("if_preds") is not None:
        if_preds_live = st.session_state["if_preds"]
        n_if = int(if_preds_live.sum())
        n_if_only = int(((if_preds_live == 1) & ((probas_live >= threshold) == False)).sum())
        n_total = int(((probas_live >= threshold) | (if_preds_live == 1)).sum())
        st.sidebar.metric("Alertes totales", f"{n_total}")
        st.sidebar.caption(f"RF : {n_alerts_rf} | IF : +{n_if_only}")
    elif use_if:
        st.sidebar.metric("Alertes RF", f"{n_alerts_rf}")
        st.sidebar.caption("IF active — allez sur Vue d'ensemble pour calculer")
    else:
        st.sidebar.metric("Alertes", f"{n_alerts_rf}")

# Corrections utilisateur
n_corrections = len(st.session_state.get("user_corrections", {}))
if n_corrections > 0:
    st.sidebar.caption(f"Corrections manuelles : {n_corrections}")

st.sidebar.divider()
st.sidebar.caption("Micro-SIEM v5.0 | Loris Dietrich")

# === ROUTING ===
if page == "Vue d'ensemble":
    from pages_app.overview import render
    render(models, session_features, config)

elif page == "Test externe":
    from pages_app.external_test import render
    render(models, session_features, config)

elif page == "Analyse detaillee":
    from pages_app.detail import render
    render(models, session_features, config)

elif page == "Mode cascade":
    from pages_app.cascade import render
    render(models, session_features, packet_features, config)

elif page == "Projection":
    from pages_app.visualization import render
    render(models, session_features, config)

elif page == "SHAP Global":
    from pages_app.shap_global import render
    render(models, session_features, config)

elif page == "Faux negatifs":
    from pages_app.false_negatives import render
    render(models, session_features, config)

elif page == "Clustering":
    from pages_app.clustering import render
    render(models, session_features, config)

elif page == "Methodologie":
    from pages_app.methodology import render
    render()

elif page == "Configuration":
    from pages_app.config import render
    new_config = render(config)
    if new_config:
        config = new_config

elif page == "Statistiques":
    from pages_app.stats import render
    render(models, session_features, config)

elif page == "A propos":
    from pages_app.about import render
    render()
