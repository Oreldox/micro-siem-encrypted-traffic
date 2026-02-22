"""
Page 1 : Vue d'ensemble — Import CSV/PCAP, classification, metriques, table des alertes.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card
from src.models import DEMO_DATA_PATH


def render(models, session_features, config):
    st.markdown("""
    <div class="hero-banner">
        <h1>Micro-SIEM &mdash; Classification du trafic chiffre</h1>
        <p>Cet outil analyse des sessions reseau et detecte le trafic malveillant (C2, exfiltration, ransomware)
        a partir des metadonnees de connexion, <strong>sans dechiffrer le contenu</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Upload ou demo ---
    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Importer un fichier CSV de sessions reseau",
            type=["csv", "pcap"],
            help="CSV avec les 27 features session-based. Colonne 'label' optionnelle (0=benin, 1=malveillant)."
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        load_demo = st.button("Charger la demo (5 000 sessions)",
                              use_container_width=True,
                              type="primary")

    # Charger les donnees
    if uploaded is not None:
        if uploaded.name.endswith(".pcap") or uploaded.name.endswith(".pcapng"):
            _handle_pcap_upload(uploaded, session_features)
        else:
            df = pd.read_csv(uploaded, low_memory=False)
            # Detection de format et adaptation si necessaire
            df = _handle_csv_format(df, session_features)
            if df is not None:
                st.session_state["data"] = df
                st.session_state["data_source"] = uploaded.name
    elif load_demo:
        if os.path.exists(DEMO_DATA_PATH):
            df = pd.read_csv(DEMO_DATA_PATH, low_memory=False)
            st.session_state["data"] = df
            st.session_state["data_source"] = "Demonstration : 5 000 sessions du dataset CIC-Darknet2020"
        else:
            st.error("Fichier de demonstration introuvable.")
            return

    if "data" not in st.session_state:
        _render_landing(models)
        return

    df = st.session_state["data"]
    source = st.session_state.get("data_source", "")
    st.caption(f"Source : {source}")

    # --- Predictions ---
    if "rf_session" not in models:
        st.error("Modele Random Forest non disponible.")
        return

    missing = [f for f in session_features if f not in df.columns]
    if missing:
        st.error(f"Features manquantes dans le CSV : {missing[:5]}...")
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
        render_metric_card("Alertes (suspectes)", f"{n_alerts:,}", "red" if n_alerts > 0 else "green")
    with col4:
        render_metric_card("Alertes critiques (P>0.8)", f"{n_high:,}", "yellow")

    explain(
        f"Le modele a analyse <strong>{n_total:,} sessions</strong> reseau. "
        f"Parmi elles, <strong>{n_alerts:,}</strong> ont une probabilite de malveillance "
        f"superieure au seuil de <strong>{threshold}</strong> (configurable dans l'onglet Configuration). "
        f"Les alertes critiques sont celles avec une probabilite > 0.8."
    )

    # --- Distribution des probabilites ---
    st.subheader("Distribution des probabilites")

    if has_labels:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probas[y_true == 0], nbinsx=50, name="Sessions benignes (label=0)",
            marker_color="#3b82f6", opacity=0.6
        ))
        fig.add_trace(go.Histogram(
            x=probas[y_true == 1], nbinsx=50, name="Sessions malveillantes (label=1)",
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
        "Chaque session recoit un <strong>score entre 0 et 1</strong>. "
        "Un score proche de 0 = probablement benin. Proche de 1 = probablement malveillant. "
        "La ligne rouge verticale est le seuil de decision : tout ce qui est a droite est signale comme alerte. "
        "Un bon modele separe bien les deux groupes (pics distincts a gauche et a droite)."
    )

    # --- Table des sessions a risque ---
    st.subheader("Top 100 des sessions les plus suspectes")

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

    explain(
        "Ce tableau montre les 100 sessions avec la probabilite de malveillance la plus elevee. "
        "La colonne <strong>probabilite</strong> est coloree du vert (benin) au rouge (malveillant). "
        "Si les donnees contiennent des labels, la colonne <strong>verite terrain</strong> montre "
        "la vraie nature de la session pour verifier les predictions."
    )

    # --- Export buttons ---
    _render_export_buttons(df, probas, preds_combined, session_features, config)


def _handle_csv_format(df, session_features):
    """Detecte le format du CSV et adapte si necessaire."""
    from src.feature_extraction import detect_dataset_format, adapt_dataframe

    fmt, matched, missing = detect_dataset_format(df, session_features)

    if fmt == "CIC-Darknet2020" and not missing:
        return df

    if fmt == "Format inconnu":
        st.warning(
            f"Format CSV non reconnu. Aucune des 27 features attendues n'a ete trouvee. "
            f"Colonnes du fichier : {', '.join(df.columns[:10])}..."
        )
        st.info("Utilisez un CSV au format CIC-Darknet2020 ou un fichier PCAP.")
        return None

    # Format partiellement compatible
    st.warning(
        f"Format detecte : **{fmt}**. "
        f"{len(matched)}/{len(session_features)} features trouvees. "
        f"Features manquantes (mises a 0) : {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
    )

    df_adapted = adapt_dataframe(df, session_features)
    return df_adapted


def _handle_pcap_upload(uploaded, session_features):
    """Gere l'upload d'un fichier PCAP."""
    try:
        from src.feature_extraction import extract_sessions_from_pcap, compute_session_features
        with st.spinner("Extraction des sessions depuis le fichier PCAP..."):
            pcap_bytes = uploaded.read()
            sessions_dict = extract_sessions_from_pcap(pcap_bytes)
            if not sessions_dict:
                st.error("Aucune session TCP/UDP trouvee dans le fichier PCAP.")
                return
            df = compute_session_features(sessions_dict, session_features)
            st.session_state["data"] = df
            st.session_state["data_source"] = f"PCAP : {uploaded.name} ({len(sessions_dict)} sessions extraites)"
            st.success(f"{len(sessions_dict)} sessions extraites depuis {uploaded.name}")
    except ImportError:
        st.error("Module d'extraction PCAP non disponible. Installez dpkt : `pip install dpkt`")
    except Exception as e:
        st.error(f"Erreur lors du parsing PCAP : {e}")


def _render_landing(models):
    """Affiche la page d'accueil quand aucune donnee n'est chargee."""
    from src.models import load_models

    st.markdown("---")
    st.markdown("### Comment ca marche ?")
    st.markdown("""
    1. **Importez un CSV** de sessions reseau (ou cliquez sur "Charger la demo")
    2. Le modele **Random Forest** (99.5% accuracy) analyse chaque session
    3. Les sessions suspectes sont signalees avec leur **probabilite de malveillance**
    4. Explorez les alertes dans les autres onglets (SHAP, configuration du seuil, statistiques)
    """)

    st.markdown("### Modeles pre-charges")
    _, model_info = load_models()
    cols = st.columns(3)
    descriptions = [
        "Classification principale — analyse les 27 features de chaque session",
        "Utilisee pour expliquer les predictions (SHAP waterfall)",
        "Detection non supervisee — signale les sessions anormales"
    ]
    for i, (name, size, status) in enumerate(model_info):
        with cols[i]:
            icon = "Charge" if status == "Charge" else "Non trouve"
            color = "green" if status == "Charge" else "red"
            render_metric_card(name, icon, color)
            if size != "-":
                st.caption(descriptions[i])


def _render_export_buttons(df, probas, preds, session_features, config):
    """Affiche les boutons d'export CSV et PDF."""
    st.markdown("---")
    st.subheader("Exporter les resultats")

    explain(
        "Telechargez les resultats de l'analyse au format CSV (donnees brutes) "
        "ou PDF (rapport complet avec graphiques)."
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            from src.report import generate_csv_report
            csv_data = generate_csv_report(df, probas, preds, session_features)
            st.download_button(
                "Telecharger CSV",
                data=csv_data,
                file_name="micro_siem_resultats.csv",
                mime="text/csv",
                use_container_width=True
            )
        except ImportError:
            st.button("Telecharger CSV", disabled=True, use_container_width=True)

    with col2:
        try:
            from src.report import generate_pdf_report
            y_true = st.session_state.get("y_true")
            pdf_data = generate_pdf_report(df, probas, preds, config, session_features, y_true)
            st.download_button(
                "Telecharger PDF",
                data=pdf_data,
                file_name="micro_siem_rapport.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except ImportError:
            st.button("Telecharger PDF", disabled=True, use_container_width=True)
        except Exception as e:
            st.warning(f"Erreur generation PDF : {e}")
