"""
Page 1 : Vue d'ensemble — Landing page, import CSV/PCAP, classification, metriques, alertes.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card, load_demo_data
from src.models import DEMO_DATA_PATH


def render(models, session_features, config):
    # Si aucune donnee, afficher la landing page
    if "data" not in st.session_state or "probas" not in st.session_state:
        _render_landing_page()
        return

    # Sinon, afficher les resultats d'analyse
    _render_analysis(models, session_features, config)


# =============================================================================
# LANDING PAGE
# =============================================================================

def _render_landing_page():
    """Page d'accueil quand aucune donnee n'est chargee."""

    st.markdown("""
    <div class="hero-banner">
        <h1>Micro-SIEM &mdash; Trafic Chiffre</h1>
        <p>Detectez le trafic reseau malveillant (C2, exfiltration, ransomware)
        a partir des metadonnees de connexion, <strong>sans dechiffrer le contenu</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Explication ---
    st.markdown("### Comment ca marche ?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Importez vos donnees**

        Un fichier CSV de sessions reseau
        ou une capture PCAP. Le dashboard
        detecte automatiquement le format.
        """)
    with col2:
        st.markdown("""
        **2. Classification automatique**

        Le modele Random Forest (99.5% accuracy)
        analyse chaque session et attribue une
        probabilite de malveillance.
        """)
    with col3:
        st.markdown("""
        **3. Explorez les resultats**

        Explications SHAP, mode cascade,
        projection UMAP, export PDF.
        Chaque alerte est justifiee.
        """)

    st.markdown("---")

    # --- Deux gros boutons ---
    st.markdown("### Commencer l'analyse")

    col_demo, col_import = st.columns(2)

    with col_demo:
        st.markdown("""
        <div class="metric-card">
            <h3>Donnees de demonstration</h3>
            <div class="value blue" style="font-size:1.2rem">5 000 sessions CIC-Darknet2020</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Sessions reseau labelisees (benin / malveillant) pour explorer toutes les fonctionnalites du dashboard.")
        if st.button("Utiliser les donnees de demo", type="primary", use_container_width=True):
            load_demo_data()
            st.rerun()

    with col_import:
        st.markdown("""
        <div class="metric-card">
            <h3>Vos propres donnees</h3>
            <div class="value green" style="font-size:1.2rem">CSV ou PCAP</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Importez vos sessions reseau (CSV 27 features) ou une capture reseau (.pcap) pour une analyse reelle.")
        uploaded = st.file_uploader(
            "Importer",
            type=["csv", "pcap"],
            label_visibility="collapsed",
            help="CSV session-based (27 features) ou fichier PCAP."
        )
        if uploaded is not None:
            _process_upload(uploaded)
            st.rerun()

    st.markdown("---")

    # --- Modeles disponibles ---
    st.markdown("### Modeles pre-entraines")

    from src.models import load_models
    _, model_info = load_models()

    col1, col2, col3 = st.columns(3)
    descriptions = [
        "Classification principale — analyse les 27 features de chaque session (F1 = 0.995)",
        "Explications SHAP — comprendre pourquoi le modele classe une session (F1 = 0.990)",
        "Detection non supervisee — signale les sessions anormales (Precision = 94.7%)"
    ]
    for i, (name, size, status) in enumerate(model_info):
        with [col1, col2, col3][i]:
            icon = "OK" if status == "Charge" else "Non trouve"
            color = "green" if status == "Charge" else "red"
            render_metric_card(name, icon, color)
            st.caption(descriptions[i])


def _process_upload(uploaded):
    """Traite un fichier uploade (CSV ou PCAP)."""
    from src.models import load_models, load_feature_mapping, SESSION_MAPPING_PATH
    session_features = load_feature_mapping(SESSION_MAPPING_PATH)

    if uploaded.name.endswith(".pcap") or uploaded.name.endswith(".pcapng"):
        try:
            from src.feature_extraction import extract_sessions_from_pcap, compute_session_features
            pcap_bytes = uploaded.read()
            sessions_dict = extract_sessions_from_pcap(pcap_bytes)
            if not sessions_dict:
                st.error("Aucune session TCP/UDP trouvee dans le PCAP.")
                return
            df = compute_session_features(sessions_dict, session_features)
            st.session_state["data"] = df
            st.session_state["data_source"] = f"PCAP : {uploaded.name} ({len(sessions_dict)} sessions)"
        except Exception as e:
            st.error(f"Erreur PCAP : {e}")
            return
    else:
        df = pd.read_csv(uploaded, low_memory=False)
        from src.feature_extraction import detect_dataset_format, adapt_dataframe
        fmt, matched, missing = detect_dataset_format(df, session_features)
        if fmt == "Format inconnu":
            st.error("Format CSV non reconnu. Utilisez un CSV CIC-Darknet2020 ou un fichier PCAP.")
            return
        if missing:
            df = adapt_dataframe(df, session_features)
        st.session_state["data"] = df
        st.session_state["data_source"] = uploaded.name

    # Lancer les predictions
    models, _ = load_models()
    if "rf_session" in models and "data" in st.session_state:
        df = st.session_state["data"]
        avail = [f for f in session_features if f in df.columns]
        if len(avail) == len(session_features):
            X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)
            probas = models["rf_session"].predict_proba(X)[:, 1]
            preds = (probas >= 0.5).astype(int)
            st.session_state["probas"] = probas
            st.session_state["preds"] = preds
            st.session_state["preds_rf"] = preds
            st.session_state["if_preds"] = None
            st.session_state["if_scores"] = None
            st.session_state["X"] = X
            if "label" in df.columns:
                st.session_state["y_true"] = df["label"].values.astype(int)


# =============================================================================
# ANALYSE (quand des donnees sont chargees)
# =============================================================================

def _render_analysis(models, session_features, config):
    """Affiche les resultats d'analyse."""

    df = st.session_state["data"]
    source = st.session_state.get("data_source", "")

    # Header compact
    col_title, col_source, col_new = st.columns([3, 4, 1])
    with col_title:
        st.markdown("## Vue d'ensemble")
    with col_source:
        st.caption(f"Source : {source}")
    with col_new:
        if st.button("Nouvelle analyse", use_container_width=True):
            for key in ["data", "data_source", "probas", "preds", "preds_rf",
                        "if_preds", "if_scores", "X", "y_true", "packet_data",
                        "packet_source", "umap_embedding", "umap_indices"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Recalculer les predictions avec le seuil actuel
    if "rf_session" not in models:
        st.error("Modele Random Forest non disponible.")
        return

    missing = [f for f in session_features if f not in df.columns]
    if missing:
        st.error(f"Features manquantes : {missing[:5]}...")
        return

    X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)
    model_rf = models["rf_session"]

    probas = model_rf.predict_proba(X)[:, 1]
    threshold = config["threshold"]
    preds = (probas >= threshold).astype(int)

    # Scores IF si active
    if_scores = None
    if_preds = None
    if config["use_if"] and "isolation_forest" in models:
        if_raw = models["isolation_forest"].predict(X)
        if_preds = (if_raw == -1).astype(int)
        if_scores = models["isolation_forest"].decision_function(X)
        preds_combined = ((preds == 1) | (if_preds == 1)).astype(int)
    else:
        preds_combined = preds

    # Sauvegarder en session_state
    st.session_state["probas"] = probas
    st.session_state["preds"] = preds_combined
    st.session_state["preds_rf"] = preds
    st.session_state["if_preds"] = if_preds
    st.session_state["if_scores"] = if_scores
    st.session_state["X"] = X

    has_labels = "label" in df.columns
    if has_labels:
        y_true = df["label"].values.astype(int)
        st.session_state["y_true"] = y_true

    # --- Metriques ---
    n_total = len(df)
    n_alerts = int(preds_combined.sum())
    n_high = int((probas >= 0.8).sum())
    n_benign = n_total - n_alerts

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Sessions analysees", f"{n_total:,}", "blue")
    with col2:
        render_metric_card("Sessions benignes", f"{n_benign:,}", "green")
    with col3:
        render_metric_card("Alertes", f"{n_alerts:,}", "red" if n_alerts > 0 else "green")
    with col4:
        render_metric_card("Critiques (P>0.8)", f"{n_high:,}", "yellow")

    explain(
        f"<strong>{n_total:,} sessions</strong> analysees. "
        f"<strong>{n_alerts:,}</strong> depassent le seuil de {threshold} (configurable dans la sidebar). "
        f"Les alertes critiques (P > 0.8) sont les plus suspectes."
    )

    # --- Distribution des probabilites ---
    st.subheader("Distribution des probabilites")

    if has_labels:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probas[y_true == 0], nbinsx=50, name="Benignes (label=0)",
            marker_color="#3b82f6", opacity=0.6
        ))
        fig.add_trace(go.Histogram(
            x=probas[y_true == 1], nbinsx=50, name="Malveillantes (label=1)",
            marker_color="#ef4444", opacity=0.6
        ))
        fig.update_layout(barmode="overlay")
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probas, nbinsx=50, name="Toutes les sessions",
            marker_color="#3b82f6", opacity=0.7
        ))

    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil = {threshold}")
    fig.update_layout(
        xaxis_title="Probabilite de malveillance (0 = benin, 1 = malveillant)",
        yaxis_title="Nombre de sessions",
        template="plotly_dark", height=350, margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    explain(
        "Score de 0 a 1 pour chaque session. La ligne rouge = seuil de decision. "
        "A droite du seuil = alerte. Un bon modele separe bien les deux groupes."
    )

    # --- Table des sessions a risque ---
    st.subheader("Sessions les plus suspectes")

    df_display = df.copy()
    df_display["probabilite"] = probas
    df_display["verdict"] = np.where(preds_combined == 1, "SUSPECT", "Benin")
    if if_preds is not None:
        df_display["anomalie_IF"] = np.where(if_preds == 1, "OUI", "-")
    if has_labels:
        df_display["verite_terrain"] = np.where(y_true == 1, "Malveillant", "Benin")

    display_cols = ["probabilite", "verdict"]
    if if_preds is not None:
        display_cols.append("anomalie_IF")
    if has_labels:
        display_cols.append("verite_terrain")
    top_features = session_features[:5]
    display_cols.extend([f for f in top_features if f in df_display.columns])

    df_sorted = df_display[display_cols].sort_values("probabilite", ascending=False)

    st.dataframe(
        df_sorted.head(100).style.background_gradient(
            subset=["probabilite"], cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True,
        height=400
    )

    # --- Export buttons ---
    st.markdown("---")
    st.subheader("Exporter les resultats")

    col1, col2 = st.columns(2)

    with col1:
        try:
            from src.report import generate_csv_report
            csv_data = generate_csv_report(df, probas, preds_combined, session_features)
            st.download_button(
                "Telecharger CSV",
                data=csv_data,
                file_name="micro_siem_resultats.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception:
            pass

    with col2:
        try:
            from src.report import generate_pdf_report
            y_true_val = st.session_state.get("y_true")
            pdf_data = generate_pdf_report(df, probas, preds_combined, config, session_features, y_true_val)
            st.download_button(
                "Telecharger PDF",
                data=pdf_data,
                file_name="micro_siem_rapport.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception:
            pass
