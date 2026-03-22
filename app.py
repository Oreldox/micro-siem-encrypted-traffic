"""
Analyse Trafic Chiffre — Classification du trafic reseau chiffre par ML.
Point d'entree principal.
"""

import streamlit as st

st.set_page_config(
    page_title="Analyse Trafic Chiffre",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ui_components import inject_css
from src.models import (load_models, load_feature_mapping,
                        SESSION_MAPPING_PATH, PACKET_MAPPING_PATH)

inject_css()

models, model_info = load_models()
session_features = load_feature_mapping(SESSION_MAPPING_PATH)
packet_features = load_feature_mapping(PACKET_MAPPING_PATH)

if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5
if "use_if" not in st.session_state:
    st.session_state["use_if"] = False

config = {
    "threshold": st.session_state["threshold"],
    "use_if": st.session_state["use_if"],
}


# === Pages ===

def page_accueil():
    from pages_app.overview import render
    render()


def page_tester():
    from pages_app.test_model import render
    render(models, session_features, config)


def page_methodologie():
    from pages_app.methodology import render
    render()


def page_stats():
    from pages_app.stats import render
    render(models, session_features, config)


def page_visualization():
    from pages_app.visualization import render
    render(models, session_features, config)


def page_detail():
    from pages_app.detail import render
    render(models, session_features, config)


def page_cascade():
    from pages_app.cascade import render
    render(models, session_features, packet_features, config)


def page_comparison():
    from pages_app.comparison import render
    render()


def page_config():
    from pages_app.config import render
    render(config)


def page_about():
    from pages_app.about import render
    render()


pages = {
    "Presentation": [
        st.Page(page_accueil, title="Accueil", icon=":material/home:", default=True),
        st.Page(page_methodologie, title="Methodologie", icon=":material/school:"),
    ],
    "Analyser": [
        st.Page(page_tester, title="Tester le modele", icon=":material/science:"),
        st.Page(page_detail, title="Detail d'une session", icon=":material/search:"),
        st.Page(page_stats, title="Statistiques", icon=":material/bar_chart:"),
        st.Page(page_visualization, title="Projection 2D", icon=":material/scatter_plot:"),
        st.Page(page_cascade, title="Mode cascade", icon=":material/layers:"),
        st.Page(page_comparison, title="Comparer les datasets", icon=":material/compare:"),
    ],
    "Parametres": [
        st.Page(page_config, title="Configuration", icon=":material/settings:"),
        st.Page(page_about, title="A propos", icon=":material/info:"),
    ],
}

pg = st.navigation(pages)

st.sidebar.divider()
st.sidebar.caption("Analyse Trafic Chiffre | Loris Dietrich")

pg.run()
