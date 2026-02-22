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
st.sidebar.markdown('<div class="sidebar-title">Micro-SIEM</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Classification du trafic chiffre</div>',
                    unsafe_allow_html=True)

PAGES = [
    "\U0001f4ca Vue d'ensemble",
    "\U0001f50d Analyse detaillee",
    "\U0001f500 Mode cascade",
    "\U0001f30c Projection UMAP",
    "\u2699\ufe0f Configuration",
    "\U0001f4c8 Statistiques",
    "\u2139\ufe0f A propos"
]

page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.divider()

# Config rapide dans la sidebar
st.sidebar.subheader("Parametres rapides")
threshold = st.sidebar.slider("Seuil de detection", 0.0, 1.0, 0.5, 0.01,
                              key="sidebar_threshold",
                              help="Sessions avec P(malveillant) >= seuil = alerte")
use_if = st.sidebar.toggle("Isolation Forest", False, key="sidebar_if",
                           help="Activer la detection d'anomalies non supervisee")

config = {"threshold": threshold, "use_if": use_if}

st.sidebar.divider()

# Info modeles
st.sidebar.subheader("Modeles charges")
for name, size, status in model_info:
    icon = "\u2705" if status == "Charge" else "\u274c"
    st.sidebar.caption(f"{icon} {name} ({size})")

st.sidebar.divider()
st.sidebar.caption("Micro-SIEM v2.0 | Loris Dietrich")

# === ROUTING ===
if page == PAGES[0]:  # Vue d'ensemble
    from pages.overview import render
    render(models, session_features, config)

elif page == PAGES[1]:  # Analyse detaillee
    from pages.detail import render
    render(models, session_features, config)

elif page == PAGES[2]:  # Mode cascade
    from pages.cascade import render
    render(models, session_features, packet_features, config)

elif page == PAGES[3]:  # Projection UMAP
    from pages.visualization import render
    render(models, session_features, config)

elif page == PAGES[4]:  # Configuration
    from pages.config import render
    new_config = render(config)
    if new_config:
        config = new_config

elif page == PAGES[5]:  # Statistiques
    from pages.stats import render
    render(models, session_features, config)

elif page == PAGES[6]:  # A propos
    from pages.about import render
    render()
