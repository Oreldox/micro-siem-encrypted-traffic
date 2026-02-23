"""
Page 2 : Analyse detaillee — Explication SHAP par session, comparaison features.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.ui_components import explain, render_metric_card
from src.models import load_shap_explainer


def render(models, session_features, config):
    st.header("Analyse detaillee d'une session")
    st.markdown("Selectionnez une session pour comprendre **pourquoi** le modele la considere comme suspecte ou benigne.")

    from src.ui_components import require_data
    if not require_data("Selectionnez une session pour voir l'explication SHAP de sa classification."):
        return

    df = st.session_state["data"]
    probas = st.session_state["probas"]
    preds = st.session_state["preds"]
    X = st.session_state["X"]

    # --- Selection de session ---
    st.markdown("---")
    n = len(df)
    col_select, col_filter = st.columns([2, 1])
    with col_filter:
        filter_type = st.radio("Filtrer par", ["Toutes", "Alertes uniquement", "Top 50 suspectes"],
                               horizontal=True)
    with col_select:
        if filter_type == "Alertes uniquement":
            alert_indices = np.where(preds == 1)[0]
            if len(alert_indices) == 0:
                st.info("Aucune alerte avec le seuil actuel. Baissez le seuil dans Configuration.")
                return
            idx = st.selectbox("Choisir une session", alert_indices,
                               format_func=lambda i: f"Session {i} — probabilite {probas[i]:.4f}")
        elif filter_type == "Top 50 suspectes":
            top_indices = np.argsort(probas)[::-1][:50]
            idx = st.selectbox("Choisir une session", top_indices,
                               format_func=lambda i: f"Session {i} — probabilite {probas[i]:.4f}")
        else:
            idx = st.number_input("Index de session", min_value=0, max_value=n-1, value=0)

    # --- Verdict ---
    proba = probas[idx]
    verdict = "SUSPECT" if preds[idx] == 1 else "Benin"

    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        color = "red" if proba >= 0.5 else ("yellow" if proba >= 0.3 else "green")
        render_metric_card("Probabilite de malveillance", f"{proba:.4f}", color)
    with col_v2:
        render_metric_card("Verdict du modele", verdict, "red" if verdict == "SUSPECT" else "green")
    with col_v3:
        if "y_true" in st.session_state:
            label = "Malveillant" if st.session_state["y_true"][idx] == 1 else "Benin"
            correct = (st.session_state["y_true"][idx] == preds[idx])
            render_metric_card("Verite terrain", label, "green" if correct else "red")
        else:
            render_metric_card("Verite terrain", "Non disponible", "blue")

    st.markdown("---")

    # --- SHAP explanation ---
    if "xgboost" in models:
        st.subheader("Pourquoi cette prediction ? (Explication SHAP)")

        explain(
            "<strong>SHAP</strong> decompose la prediction en contributions de chaque feature. "
            "Chaque barre montre l'influence d'une caracteristique sur la decision : "
            "<span style='color:#ef4444'>en rouge</span> = pousse vers 'malveillant', "
            "<span style='color:#3b82f6'>en bleu</span> = pousse vers 'benin'. "
            "Les features en haut sont celles qui ont le plus d'impact sur cette session."
        )

        try:
            import shap
            import matplotlib.pyplot as plt

            explainer = load_shap_explainer(models["xgboost"])
            X_session = X[idx:idx+1]
            shap_values = explainer.shap_values(X_session)

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_session[0],
                feature_names=session_features
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Erreur SHAP : {e}")
    else:
        st.info("Modele XGBoost non disponible pour les explications SHAP.")

    st.markdown("---")

    # --- Comparaison feature values ---
    st.subheader("Valeurs des features de cette session")

    explain(
        "Ce tableau compare les valeurs de la session selectionnee avec la moyenne du dataset. "
        "Le <strong>Z-score</strong> mesure l'ecart par rapport a la normale : "
        "un Z-score > 2 (rouge) signifie que la valeur est anormalement elevee, "
        "< -2 (bleu) = anormalement basse. "
        "C'est un indicateur de ce qui rend cette session differente des autres."
    )

    feature_vals = X[idx]
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)

    z_scores = np.where(std_all > 0, (feature_vals - mean_all) / std_all, 0)

    df_features = pd.DataFrame({
        "Feature": session_features,
        "Valeur (cette session)": feature_vals,
        "Moyenne (dataset)": mean_all,
        "Ecart-type": std_all,
        "Z-score": z_scores
    }).sort_values("Z-score", key=abs, ascending=False)

    st.dataframe(
        df_features.style.background_gradient(
            subset=["Z-score"], cmap="RdBu_r", vmin=-3, vmax=3
        ).format({
            "Valeur (cette session)": "{:.4f}",
            "Moyenne (dataset)": "{:.4f}",
            "Ecart-type": "{:.4f}",
            "Z-score": "{:.2f}"
        }),
        use_container_width=True,
        height=500
    )
