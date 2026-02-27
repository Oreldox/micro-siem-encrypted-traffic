"""
Analyse Trafic Chiffre V5 — Classification du trafic reseau chiffre par ML.
Point d'entree principal : routing, sidebar, configuration globale.
"""

import streamlit as st

# === PAGE CONFIG (doit etre le premier appel Streamlit) ===
st.set_page_config(
    page_title="Analyse Trafic Chiffre",
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

# === CONFIG VIA SESSION_STATE ===
if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5
if "use_if" not in st.session_state:
    st.session_state["use_if"] = False

config = {
    "threshold": st.session_state["threshold"],
    "use_if": st.session_state["use_if"],
}

# === SIDEBAR ===
st.sidebar.markdown('<div class="sidebar-title">\U0001f6e1\ufe0f Analyse Trafic Chiffre</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Detection de trafic malveillant par ML</div>',
                    unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "Accueil",
    "Test externe",
    "Analyse detaillee",
    "Mode cascade",
    "Projection",
    "Methodologie",
    "SHAP Global",
    "Faux negatifs",
    "Clustering",
    "Statistiques",
    "Configuration",
    "A propos"
], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Analyse Trafic Chiffre v5.0 | Loris Dietrich")

# === ROUTING ===
if page == "Accueil":
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

elif page == "Methodologie":
    from pages_app.methodology import render
    render()

elif page == "SHAP Global":
    from pages_app.shap_global import render
    render(models, session_features, config)

elif page == "Faux negatifs":
    from pages_app.false_negatives import render
    render(models, session_features, config)

elif page == "Clustering":
    from pages_app.clustering import render
    render(models, session_features, config)

elif page == "Statistiques":
    from pages_app.stats import render
    render(models, session_features, config)

elif page == "Configuration":
    from pages_app.config import render
    render(config)

elif page == "A propos":
    from pages_app.about import render
    render()
