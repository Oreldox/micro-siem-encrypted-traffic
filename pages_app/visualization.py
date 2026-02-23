"""
Page 4 : Projection UMAP — Visualisation 2D des clusters de sessions.
"""

import os
import streamlit as st
import numpy as np

from src.ui_components import explain, render_metric_card
from src.models import UMAP_EMBEDDING_PATH
from src.projection import load_precomputed_embedding, compute_umap_embedding, create_projection_figure


def render(models, session_features, config):
    st.header("Projection UMAP des sessions")

    explain(
        "<strong>UMAP</strong> (Uniform Manifold Approximation and Projection) projette les sessions "
        "en 2 dimensions tout en preservant la structure des donnees. "
        "Les sessions similaires se retrouvent proches, formant des <strong>clusters</strong> visuels. "
        "Cela permet d'identifier des familles de malwares et de reperer les faux negatifs."
    )

    # --- Choix de la source ---
    st.markdown("---")

    has_data = "probas" in st.session_state and "X" in st.session_state
    has_precomputed = os.path.exists(UMAP_EMBEDDING_PATH)

    if has_precomputed:
        source = st.radio(
            "Source de la projection",
            ["Embedding pre-calcule (demo, instantane)", "Calculer a la volee (donnees chargees)"],
            horizontal=True,
            help="L'embedding pre-calcule utilise les 5000 sessions de demo. Le calcul a la volee prend 30-60s."
        )
        use_precomputed = source.startswith("Embedding")
    else:
        use_precomputed = False
        if not has_data:
            st.warning("Chargez des donnees dans **Vue d'ensemble** pour utiliser la projection UMAP.")
            return

    # --- Charger ou calculer l'embedding ---
    if use_precomputed:
        embedding, labels = load_precomputed_embedding(UMAP_EMBEDDING_PATH)

        # On a besoin des probas pour la colorisation — charger les donnees demo si pas deja fait
        if has_data:
            probas = st.session_state["probas"]
            preds = st.session_state["preds"]
            if len(probas) != len(embedding):
                probas = None
                preds = None
        else:
            probas = None
            preds = None

        n_points = len(embedding)
        st.caption(f"Embedding pre-calcule : {n_points} sessions")

    else:
        if not has_data:
            st.warning("Chargez des donnees dans **Vue d'ensemble** d'abord.")
            return

        X = st.session_state["X"]
        probas = st.session_state["probas"]
        preds = st.session_state["preds"]
        labels = st.session_state.get("y_true")

        n_points = len(X)
        if n_points > 10000:
            st.warning(f"Les donnees contiennent {n_points:,} sessions. UMAP sera calcule sur un echantillon de 10 000.")

        if st.button("Calculer la projection UMAP", type="primary"):
            with st.spinner(f"Calcul UMAP sur {min(n_points, 10000)} sessions... (30-60 secondes)"):
                embedding, indices = compute_umap_embedding(X, max_samples=10000)
                st.session_state["umap_embedding"] = embedding
                st.session_state["umap_indices"] = indices
        elif "umap_embedding" in st.session_state:
            embedding = st.session_state["umap_embedding"]
            indices = st.session_state["umap_indices"]
        else:
            st.info("Cliquez sur le bouton pour lancer le calcul UMAP.")
            return

        # Sous-echantillonner les labels/probas/preds si necessaire
        if "umap_indices" in st.session_state:
            indices = st.session_state["umap_indices"]
            if labels is not None:
                labels = labels[indices]
            probas = probas[indices]
            preds = preds[indices]
        n_points = len(embedding)

    # --- Options de visualisation ---
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        color_options = ["label", "proba", "verdict"]
        color_labels = {
            "label": "Label reel (benin/malveillant)",
            "proba": "Probabilite RF (gradient)",
            "verdict": "Verdict du modele (seuil)"
        }
        available = []
        if labels is not None:
            available.append("label")
        if probas is not None:
            available.extend(["proba", "verdict"])
        if not available:
            available = ["label"]

        color_by = st.selectbox(
            "Coloriser par",
            available,
            format_func=lambda x: color_labels.get(x, x)
        )

    with col2:
        show_fn = False
        if labels is not None and preds is not None:
            show_fn = st.toggle("Mettre en evidence les faux negatifs", value=True)

    # --- Afficher la projection ---
    fig = create_projection_figure(
        embedding,
        labels=labels,
        preds=preds,
        probas=probas,
        color_by=color_by,
        highlight_fn=show_fn
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Statistiques ---
    if labels is not None:
        n_benin = int((labels == 0).sum())
        n_mal = int((labels == 1).sum())
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Sessions projetees", f"{n_points:,}", "blue")
        with col2:
            render_metric_card("Benignes", f"{n_benin:,}", "green")
        with col3:
            render_metric_card("Malveillantes", f"{n_mal:,}", "red")

    if labels is not None and preds is not None:
        fn_count = int(((labels == 1) & (preds == 0)).sum())
        if fn_count > 0:
            explain(
                f"<strong>{fn_count} faux negatifs</strong> sont visibles sur la projection (cercles jaunes). "
                "Ces sessions malveillantes n'ont pas ete detectees par le modele. "
                "Observer leur position aide a comprendre pourquoi : sont-elles isolees ou proches des sessions benignes ?"
            )

    explain(
        "Chaque point represente une session reseau. Les sessions proches partagent des caracteristiques similaires. "
        "Les <strong>clusters</strong> (groupes denses) correspondent souvent a des familles de trafic "
        "(navigateur normal, streaming, C2, etc.). "
        "Survolez un point pour voir ses details."
    )
