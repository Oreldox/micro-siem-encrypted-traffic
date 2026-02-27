"""
Page 4 : Projection — Visualisation 2D des clusters (PCA ou UMAP automatique).
"""

import os
import streamlit as st
import numpy as np

from src.ui_components import explain, render_metric_card, is_demo_data, has_labels
from src.models import UMAP_EMBEDDING_PATH
from src.projection import load_precomputed_embedding, compute_projection_embedding, create_projection_figure


def render(models, session_features, config):
    st.header("Projection des sessions")

    explain(
        "La projection reduit les 27 dimensions des features en <strong>2 dimensions</strong> "
        "tout en preservant la structure des donnees. "
        "Les sessions similaires se retrouvent proches, formant des <strong>clusters</strong> visuels. "
        "Methode : <strong>PCA</strong> (< 50 sessions, instantane) ou <strong>UMAP</strong> (>= 50 sessions)."
    )

    from src.ui_components import require_data
    if not require_data("Visualisez les sessions en 2D pour identifier les clusters."):
        return

    st.markdown("---")

    X = st.session_state["X"]
    probas = st.session_state["probas"]
    preds = st.session_state["preds"]
    labels = st.session_state.get("y_true")

    # --- Auto-detect : demo -> precomputed, sinon -> compute ---
    if is_demo_data() and os.path.exists(UMAP_EMBEDDING_PATH):
        embedding, embed_labels = load_precomputed_embedding(UMAP_EMBEDDING_PATH)

        if len(probas) == len(embedding):
            labels = embed_labels
        else:
            probas = None
            preds = None
            labels = embed_labels

        method = "UMAP"
        n_points = len(embedding)
        st.caption(f"Projection UMAP pre-calculee : {n_points:,} sessions (demo)")
    else:
        n = len(X)
        if n < 3:
            st.warning("Trop peu de sessions pour une projection (minimum 3).")
            return

        # Verifier le cache
        cache_key = f"projection_{n}"
        if st.session_state.get("projection_cache_key") == cache_key and "umap_embedding" in st.session_state:
            embedding = st.session_state["umap_embedding"]
            indices = st.session_state["umap_indices"]
            method = st.session_state.get("projection_method", "PCA")
        else:
            method_hint = "PCA (instantane)" if n < 50 else f"UMAP ({n:,} sessions)"
            with st.spinner(f"Calcul de la projection {method_hint}..."):
                embedding, indices, method = compute_projection_embedding(X)
                st.session_state["umap_embedding"] = embedding
                st.session_state["umap_indices"] = indices
                st.session_state["projection_method"] = method
                st.session_state["projection_cache_key"] = cache_key

        # Sous-echantillonner si necessaire
        indices = st.session_state["umap_indices"]
        if labels is not None:
            labels = labels[indices]
        probas = probas[indices]
        preds = preds[indices]
        n_points = len(embedding)
        st.caption(f"Projection {method} : {n_points:,} sessions")

    # --- Options de visualisation ---
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
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
            available = ["proba"]

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
        highlight_fn=show_fn,
        method=method
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
    else:
        render_metric_card("Sessions projetees", f"{n_points:,}", "blue")

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
