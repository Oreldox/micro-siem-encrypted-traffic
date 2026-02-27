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
from src.models import load_models, load_feature_mapping, SESSION_MAPPING_PATH

inject_css()

models, model_info = load_models()
session_features = load_feature_mapping(SESSION_MAPPING_PATH)

if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5
if "use_if" not in st.session_state:
    st.session_state["use_if"] = False

config = {
    "threshold": st.session_state["threshold"],
    "use_if": st.session_state["use_if"],
}


def page_accueil():
    from pages_app.overview import render
    render()


def page_tester():
    from pages_app.test_model import render
    render(models, session_features, config)


def page_detail():
    from pages_app.detail import render
    render(models, session_features, config)


def page_methodologie():
    from pages_app.methodology import render
    render()


def page_shap():
    from pages_app.shap_global import render
    render(models, session_features, config)


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
    "Tester le modele": [
        st.Page(page_tester, title="Tester le modele", icon=":material/science:"),
        st.Page(page_detail, title="Analyse detaillee", icon=":material/search:"),
        st.Page(page_shap, title="Explicabilite SHAP", icon=":material/insights:"),
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
