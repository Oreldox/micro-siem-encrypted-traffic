"""
Analyse Trafic Chiffre V5 — Classification du trafic reseau chiffre par ML.
Point d'entree principal : navigation groupee, configuration globale.
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


# === FONCTIONS DE PAGE ===

def page_accueil():
    from pages_app.overview import render
    render(models, session_features, config)


def page_test_externe():
    from pages_app.external_test import render
    render(models, session_features, config)


def page_detail():
    from pages_app.detail import render
    render(models, session_features, config)


def page_cascade():
    from pages_app.cascade import render
    render(models, session_features, packet_features, config)


def page_projection():
    from pages_app.visualization import render
    render(models, session_features, config)


def page_methodologie():
    from pages_app.methodology import render
    render()


def page_shap():
    from pages_app.shap_global import render
    render(models, session_features, config)


def page_fn():
    from pages_app.false_negatives import render
    render(models, session_features, config)


def page_clustering():
    from pages_app.clustering import render
    render(models, session_features, config)


def page_stats():
    from pages_app.stats import render
    render(models, session_features, config)


def page_config():
    from pages_app.config import render
    render(config)


def page_about():
    from pages_app.about import render
    render()


# === NAVIGATION GROUPEE ===
pages = {
    "Analyse du trafic": [
        st.Page(page_accueil, title="Accueil", icon=":material/home:", default=True),
        st.Page(page_test_externe, title="Test externe", icon=":material/science:"),
        st.Page(page_detail, title="Analyse detaillee", icon=":material/search:"),
        st.Page(page_cascade, title="Mode cascade", icon=":material/layers:"),
        st.Page(page_projection, title="Projection", icon=":material/scatter_plot:"),
    ],
    "Performance du modele": [
        st.Page(page_stats, title="Statistiques", icon=":material/bar_chart:"),
        st.Page(page_methodologie, title="Methodologie", icon=":material/school:"),
        st.Page(page_shap, title="Explicabilite SHAP", icon=":material/insights:"),
        st.Page(page_fn, title="Faux negatifs", icon=":material/warning:"),
        st.Page(page_clustering, title="Clustering", icon=":material/hub:"),
    ],
    "Parametres": [
        st.Page(page_config, title="Configuration", icon=":material/settings:"),
        st.Page(page_about, title="A propos", icon=":material/info:"),
    ],
}

pg = st.navigation(pages)

st.sidebar.divider()
st.sidebar.caption("Analyse Trafic Chiffre v5.0 | Loris Dietrich")

pg.run()
