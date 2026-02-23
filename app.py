"""
Micro-SIEM Dashboard V2 — Classification du trafic reseau chiffre.
Point d'entree principal : routing, sidebar, configuration globale.
"""

import streamlit as st

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

page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Analyse detaillee",
    "Mode cascade",
    "Projection UMAP",
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

st.sidebar.divider()
st.sidebar.caption("Micro-SIEM v2.0 | Loris Dietrich")

# === ROUTING ===
if page == "Vue d'ensemble":
    from pages_app.overview import render
    render(models, session_features, config)

elif page == "Analyse detaillee":
    from pages_app.detail import render
    render(models, session_features, config)

elif page == "Mode cascade":
    from pages_app.cascade import render
    render(models, session_features, packet_features, config)

elif page == "Projection UMAP":
    from pages_app.visualization import render
    render(models, session_features, config)

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
