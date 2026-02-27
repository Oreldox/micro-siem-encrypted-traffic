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
        <h1>Analyse Trafic Chiffre</h1>
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

        PCAP, PCAPNG, CSV Wireshark,
        CICFlowMeter, Excel, Zeek conn.log.
        Detection automatique du format.
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
            if load_demo_data():
                st.rerun()

    with col_import:
        st.markdown("""
        <div class="metric-card">
            <h3>Vos donnees reseau</h3>
            <div class="value green" style="font-size:1.2rem">Multi-format</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("PCAP/PCAPNG, CSV (Wireshark, CICFlowMeter, tshark), Excel, Zeek conn.log")
        uploaded = st.file_uploader(
            "Importer un fichier",
            type=["pcap", "pcapng", "csv", "xlsx", "xls", "log", "tsv"],
            label_visibility="collapsed",
            help="Formats : PCAP/PCAPNG, CSV (CIC-Darknet2020, CICFlowMeter, Wireshark, tshark), Excel, Zeek conn.log"
        )
        if uploaded is not None:
            success = _process_upload(uploaded)
            if success:
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
    """Traite un fichier uploade (multi-format). Retourne True si succes."""
    from src.models import load_models, load_feature_mapping, SESSION_MAPPING_PATH
    session_features = load_feature_mapping(SESSION_MAPPING_PATH)
    filename = uploaded.name.lower()

    # Validation : taille du fichier
    file_size = uploaded.size if hasattr(uploaded, 'size') else 0
    if file_size > 500 * 1024 * 1024:  # > 500 Mo
        st.error(
            f"Fichier trop volumineux ({file_size / (1024*1024):.0f} Mo). "
            "Limite recommandee : 500 Mo. Pour les gros fichiers PCAP, "
            "utilisez tshark pour filtrer avant l'import."
        )
        return False

    # Validation : fichier vide
    if file_size == 0:
        st.error("Le fichier est vide.")
        return False

    # =========================================================================
    # 1. PCAP / PCAPNG
    # =========================================================================
    if filename.endswith(".pcap") or filename.endswith(".pcapng"):
        try:
            from src.feature_extraction import extract_sessions_from_pcap, compute_session_features
            pcap_bytes = uploaded.read()
            sessions_dict = extract_sessions_from_pcap(pcap_bytes)
            if not sessions_dict:
                st.error("Aucune session TCP/UDP trouvee dans le fichier.")
                return False
            df = compute_session_features(sessions_dict, session_features)
            st.session_state["data"] = df
            st.session_state["data_source"] = f"PCAP : {uploaded.name} ({len(sessions_dict)} sessions)"
            st.session_state["feature_quality"] = {"total": 27, "available": 27}
        except Exception as e:
            st.error(f"Erreur PCAP : {e}")
            return False

    # =========================================================================
    # 2. ZEEK CONN.LOG (.log, .tsv avec headers Zeek)
    # =========================================================================
    elif filename.endswith(".log") or filename.endswith(".tsv"):
        try:
            content = uploaded.read().decode("utf-8", errors="replace")
            from src.feature_extraction import parse_zeek_connlog, adapt_zeek_connlog
            df_raw = parse_zeek_connlog(content)
            if df_raw is not None and "id.orig_h" in df_raw.columns:
                df = adapt_zeek_connlog(df_raw, session_features)
                n_features = sum(1 for f in session_features if f in df.columns and (df[f] != 0).any())
                st.session_state["data"] = df
                st.session_state["data_source"] = f"Zeek : {uploaded.name} ({len(df)} sessions)"
                st.session_state["feature_quality"] = {"total": 27, "available": n_features}
            else:
                # Pas un Zeek, essayer comme TSV generique
                uploaded.seek(0)
                df_raw = pd.read_csv(uploaded, sep="\t", low_memory=False, comment="#")
                return _process_tabular(df_raw, uploaded.name, session_features)
        except Exception as e:
            st.error(f"Erreur Zeek/TSV : {e}")
            return False

    # =========================================================================
    # 3. EXCEL
    # =========================================================================
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        try:
            df_raw = pd.read_excel(uploaded, engine="openpyxl" if filename.endswith(".xlsx") else None)
            return _process_tabular(df_raw, uploaded.name, session_features)
        except ImportError:
            st.error("Le module `openpyxl` est requis pour lire les fichiers Excel. Installez-le avec `pip install openpyxl`.")
            return False
        except Exception as e:
            st.error(f"Erreur Excel : {e}")
            return False

    # =========================================================================
    # 4. CSV (CIC-Darknet2020, CICFlowMeter, Wireshark packets, tshark, autre)
    # =========================================================================
    else:
        try:
            df_raw = pd.read_csv(uploaded, low_memory=False)
            return _process_tabular(df_raw, uploaded.name, session_features)
        except Exception as e:
            st.error(f"Erreur CSV : {e}")
            return False

    # Lancer les predictions
    return _run_predictions(session_features)


def _process_tabular(df_raw, filename, session_features):
    """Pipeline commun pour CSV/Excel/TSV : detection de format + adaptation."""
    # Validation : DataFrame vide
    if df_raw.empty:
        st.error("Le fichier ne contient aucune donnee (0 lignes).")
        return False
    if len(df_raw.columns) < 2:
        st.error("Le fichier ne contient pas assez de colonnes pour etre un dataset reseau.")
        return False

    from src.feature_extraction import detect_dataset_format, adapt_dataframe, adapt_cicflowmeter

    fmt, matched, missing, fmt_type = detect_dataset_format(df_raw, session_features)

    # --- CIC-Darknet2020 : deja au bon format ---
    if fmt == "CIC-Darknet2020":
        df = adapt_dataframe(df_raw, session_features) if missing else df_raw
        st.session_state["data"] = df
        st.session_state["data_source"] = f"CIC-Darknet2020 : {filename} ({len(df)} sessions)"
        st.session_state["feature_quality"] = {"total": 27, "available": len(matched)}

    # --- CICFlowMeter : mapping des colonnes ---
    elif fmt == "CICFlowMeter":
        df = adapt_cicflowmeter(df_raw, session_features)
        st.session_state["data"] = df
        st.session_state["data_source"] = f"CICFlowMeter : {filename} ({len(df)} sessions)"
        # Estimer les features non-zero
        n_features = sum(1 for f in session_features if f in df.columns and (df[f] != 0).any())
        st.session_state["feature_quality"] = {"total": 27, "available": n_features}

    # --- Wireshark / tshark packets : aggregation paquet → session ---
    elif fmt_type == "packet_level":
        from src.csv_aggregation import detect_packet_csv, aggregate_packets_to_sessions, get_tshark_command
        is_packet, col_map, missing_fields = detect_packet_csv(df_raw)
        if not is_packet:
            st.error("Erreur de detection du format paquet.")
            return False

        df, quality = aggregate_packets_to_sessions(df_raw, col_map, session_features)
        if df.empty:
            st.error("Aucune session TCP/UDP trouvee apres aggregation des paquets.")
            return False

        st.session_state["data"] = df
        st.session_state["data_source"] = f"Wireshark CSV : {filename} ({quality['sessions_count']} sessions, {quality['packets_used']} paquets)"
        st.session_state["feature_quality"] = quality

        # Avertissement si features manquantes
        if quality["missing_fields"]:
            missing_names = ", ".join(quality["missing_fields"])
            st.warning(
                f"**{quality['available']}/{quality['total']} features** disponibles. "
                f"Colonnes manquantes : {missing_names}.\n\n"
                f"Pour des predictions optimales, utilisez directement le fichier PCAP/PCAPNG "
                f"ou exportez avec tshark :\n```\n{get_tshark_command()}\n```"
            )

    # --- Compatible partiel : certaines features presentes ---
    elif fmt in ("Compatible (partiel)", "Partiellement compatible"):
        df = adapt_dataframe(df_raw, session_features)
        st.session_state["data"] = df
        st.session_state["data_source"] = f"{filename} ({len(df)} sessions, {len(matched)}/{len(session_features)} features)"
        st.session_state["feature_quality"] = {"total": 27, "available": len(matched)}
        if len(matched) < len(session_features):
            st.warning(
                f"**{len(matched)}/{len(session_features)} features** reconnues. "
                f"Les features manquantes sont mises a zero — les predictions seront moins fiables."
            )

    # --- Format inconnu ---
    else:
        st.error(
            "**Format non reconnu.** Les formats supportes sont :\n\n"
            "- **PCAP / PCAPNG** (Wireshark, tcpdump) — Recommande\n"
            "- **CSV CIC-Darknet2020** — Complet (27/27 features)\n"
            "- **CSV CICFlowMeter** — Complet (mapping automatique)\n"
            "- **CSV Wireshark** (File > Export Packet Dissections > CSV) — Partiel\n"
            "- **CSV tshark** (export avec champs etendus) — Complet\n"
            "- **Excel** (.xlsx) — Meme detection que CSV\n"
            "- **Zeek conn.log** — Partiel\n\n"
            "Verifiez que votre fichier contient des donnees de trafic reseau."
        )
        return False

    return _run_predictions(session_features)


def _run_predictions(session_features):
    """Lance les predictions ML sur les donnees chargees (RF + XGBoost + confiance)."""
    from src.models import load_models
    models, _ = load_models()

    if "rf_session" not in models or "data" not in st.session_state:
        return True

    df = st.session_state["data"]
    avail = [f for f in session_features if f in df.columns]
    if len(avail) != len(session_features):
        st.error(
            f"Features manquantes dans les donnees ({len(avail)}/{len(session_features)}). "
            f"Colonnes absentes : {[f for f in session_features if f not in df.columns][:5]}"
        )
        return False

    X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)

    # --- RF principal ---
    probas_rf = models["rf_session"].predict_proba(X)[:, 1]
    preds_rf = (probas_rf >= 0.5).astype(int)

    st.session_state["probas"] = probas_rf
    st.session_state["preds"] = preds_rf
    st.session_state["preds_rf"] = preds_rf
    st.session_state["X"] = X

    # --- XGBoost (pour accord inter-modeles et SHAP) ---
    probas_xgb = None
    if "xgboost" in models:
        try:
            probas_xgb = models["xgboost"].predict_proba(X)[:, 1]
            st.session_state["probas_xgb"] = probas_xgb
        except Exception:
            pass

    # --- IF initialise a None (sera recalcule dans _render_analysis si active) ---
    st.session_state["if_preds"] = None
    st.session_state["if_scores"] = None

    # --- Score de confiance ---
    from src.confidence import compute_confidence_scores
    fq = st.session_state.get("feature_quality")
    confidence = compute_confidence_scores(
        probas_rf, probas_xgb=probas_xgb, feature_quality=fq
    )
    st.session_state["confidence"] = confidence

    # --- Labels si disponibles ---
    if "label" in df.columns:
        st.session_state["y_true"] = df["label"].values.astype(int)

    # --- Seuil recommande selon qualite des donnees ---
    fq_avail = fq.get("available", 27) if fq else 27
    if fq_avail >= 24:
        st.session_state["recommended_threshold"] = 0.5
    elif fq_avail >= 18:
        st.session_state["recommended_threshold"] = 0.4
    else:
        st.session_state["recommended_threshold"] = 0.35

    return True


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

    # --- Recalculer la confiance avec IF si active ---
    from src.confidence import compute_confidence_scores
    fq = st.session_state.get("feature_quality")
    confidence = compute_confidence_scores(
        probas, probas_xgb=st.session_state.get("probas_xgb"),
        if_scores=if_scores, feature_quality=fq
    )
    st.session_state["confidence"] = confidence

    # --- Indicateur de qualite + confiance + seuil recommande ---
    fq_avail = fq.get("available", 27) if fq else 27
    rec_threshold = st.session_state.get("recommended_threshold", 0.5)
    mean_conf = float(confidence.mean())

    show_quality_row = fq_avail < 27 or threshold != rec_threshold or mean_conf < 0.6
    if show_quality_row:
        cols_q = st.columns(3)
        with cols_q[0]:
            fq_color = "green" if fq_avail >= 24 else ("yellow" if fq_avail >= 18 else "red")
            render_metric_card("Qualite des donnees", f"{fq_avail}/27 features", fq_color)
        with cols_q[1]:
            conf_color = "green" if mean_conf >= 0.7 else ("yellow" if mean_conf >= 0.5 else "red")
            render_metric_card("Confiance moyenne", f"{mean_conf:.0%}", conf_color)
        with cols_q[2]:
            if threshold != rec_threshold:
                render_metric_card("Seuil recommande", f"{rec_threshold}", "yellow")
            else:
                render_metric_card("Seuil", f"{threshold} (optimal)", "green")

        if fq_avail < 18:
            st.caption(
                "Predictions approximatives — utilisez un PCAP ou un export tshark "
                "complet pour de meilleurs resultats."
            )
        if threshold != rec_threshold and fq_avail < 24:
            st.caption(
                f"Le seuil recommande pour ce type de donnees est **{rec_threshold}** "
                f"(actuel : {threshold}). Ajustez dans la sidebar."
            )

    # --- Metriques ---
    n_total = len(df)
    n_alerts = int(preds_combined.sum())
    n_high = int((probas >= 0.8).sum())
    n_benign = n_total - n_alerts

    if config["use_if"] and if_preds is not None:
        n_rf_only = int(((preds == 1) & (if_preds == 0)).sum())
        n_if_only = int(((preds == 0) & (if_preds == 1)).sum())
        n_both = int(((preds == 1) & (if_preds == 1)).sum())

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_card("Sessions analysees", f"{n_total:,}", "blue")
        with col2:
            render_metric_card("Sessions benignes", f"{n_benign:,}", "green")
        with col3:
            render_metric_card("Alertes RF", f"{int(preds.sum()):,}", "red" if preds.sum() > 0 else "green")
        with col4:
            render_metric_card("Anomalies IF", f"+{n_if_only:,}", "yellow")
        with col5:
            render_metric_card("Critiques (P>0.8)", f"{n_high:,}", "red" if n_high > 0 else "green")

        explain(
            f"<strong>{n_total:,} sessions</strong> analysees. "
            f"Le Random Forest a detecte <strong>{int(preds.sum()):,}</strong> sessions suspectes (seuil {threshold}). "
            f"L'Isolation Forest a signale <strong>{n_if_only:,}</strong> sessions supplementaires comme anomalies. "
            f"Total : <strong>{n_alerts:,}</strong> alertes."
        )

        with st.expander("Detail des sources d'alertes", expanded=False):
            col_rf, col_if, col_both = st.columns(3)
            with col_rf:
                render_metric_card("RF uniquement", f"{n_rf_only:,}", "red")
            with col_if:
                render_metric_card("IF uniquement", f"{n_if_only:,}", "yellow")
            with col_both:
                render_metric_card("RF + IF", f"{n_both:,}", "red")
            explain(
                "<strong>RF uniquement</strong> : le classificateur supervise classe la session comme malveillante. "
                "<strong>IF uniquement</strong> : la session est statistiquement anormale mais le RF la considere benigne — "
                "attention particuliere requise. "
                "<strong>RF + IF</strong> : les deux modeles concordent."
            )
    else:
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

    # --- Synthese de l'analyse ---
    st.markdown("---")
    st.subheader("Synthese de l'analyse")
    _render_analysis_summary(n_total, n_alerts, n_high, probas, preds_combined, threshold, has_labels)

    # --- Distribution des probabilites ---
    st.markdown("---")
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
    conf = st.session_state.get("confidence")
    if conf is not None:
        df_display["confiance"] = conf
    if if_preds is not None:
        df_display["anomalie_IF"] = np.where(if_preds == 1, "OUI", "-")
    if has_labels:
        df_display["verite_terrain"] = np.where(y_true == 1, "Malveillant", "Benin")

    display_cols = ["probabilite", "verdict"]
    if conf is not None:
        display_cols.append("confiance")
    if if_preds is not None:
        display_cols.append("anomalie_IF")
    if has_labels:
        display_cols.append("verite_terrain")

    # Ajouter les top 5 features avec noms lisibles
    from pages_app.detail import FEATURE_NAMES_FR
    top_features = session_features[:5]
    rename_map = {}
    for f in top_features:
        if f in df_display.columns:
            fr_name = FEATURE_NAMES_FR.get(f, f)
            # Renommer pour lisibilite dans le tableau
            rename_map[f] = fr_name
            display_cols.append(f)

    df_sorted = df_display[display_cols].rename(columns=rename_map).sort_values("probabilite", ascending=False)

    styled = df_sorted.head(100).style.background_gradient(
        subset=["probabilite"], cmap="RdYlGn_r", vmin=0, vmax=1
    )
    if if_preds is not None and "anomalie_IF" in df_sorted.columns:
        def _highlight_if(row):
            if row.get("anomalie_IF") == "OUI":
                return ["background-color: rgba(245, 158, 11, 0.15)"] * len(row)
            return [""] * len(row)
        styled = styled.apply(_highlight_if, axis=1)

    st.dataframe(styled, use_container_width=True, height=400)

    # --- Export buttons ---
    st.markdown("---")
    st.subheader("Exporter les resultats")

    col1, col2 = st.columns(2)

    with col1:
        try:
            from src.report import generate_csv_report
            csv_data = generate_csv_report(
                df, probas, preds_combined, session_features,
                confidence=st.session_state.get("confidence"),
                corrections=st.session_state.get("user_corrections")
            )
            st.download_button(
                "Telecharger CSV",
                data=csv_data,
                file_name="micro_siem_resultats.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Export CSV indisponible : {e}")

    with col2:
        try:
            from src.report import generate_pdf_report
            y_true_val = st.session_state.get("y_true")
            if_preds_val = st.session_state.get("if_preds")
            pdf_data = generate_pdf_report(
                df, probas, preds_combined, config, session_features,
                y_true=y_true_val, if_preds=if_preds_val,
                confidence=st.session_state.get("confidence"),
                probas_xgb=st.session_state.get("probas_xgb"),
                feature_quality=st.session_state.get("feature_quality"),
            )
            st.download_button(
                "Telecharger PDF",
                data=pdf_data,
                file_name="micro_siem_rapport.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Export PDF indisponible : {e}")


def _render_analysis_summary(n_total, n_alerts, n_high, probas, preds, threshold, has_labels):
    """Genere une synthese textuelle de l'analyse enrichie (confiance + accord)."""

    alert_rate = n_alerts / max(n_total, 1)

    # Informations supplementaires
    confidence = st.session_state.get("confidence")
    mean_conf = float(confidence.mean()) if confidence is not None else None
    probas_xgb = st.session_state.get("probas_xgb")
    model_agreement = None
    if probas_xgb is not None:
        model_agreement = float(((probas >= 0.5) == (probas_xgb >= 0.5)).mean())

    # Evaluation globale
    if n_high == 0 and n_alerts == 0:
        st.markdown(
            "**Aucune menace detectee.** Le trafic analyse semble entierement normal. "
            "Aucune session ne depasse le seuil de detection."
        )
    elif n_high == 0 and n_alerts > 0:
        st.markdown(
            f"**{n_alerts} session(s) suspecte(s) detectee(s)** mais aucune n'est critique (P > 0.8). "
            f"Le niveau de risque est **modere**. Verifiez les sessions alertees dans l'onglet *Analyse detaillee*."
        )
    elif n_high > 0 and alert_rate < 0.1:
        st.markdown(
            f"**{n_high} session(s) critique(s) detectee(s)** sur {n_total:,} sessions analysees. "
            f"Le taux d'alerte est faible ({alert_rate:.1%}), ce qui indique des menaces **ciblees**. "
            f"Priorite : investiguer les sessions critiques (P > 0.8) dans l'onglet *Analyse detaillee*."
        )
    elif n_high > 0 and alert_rate >= 0.1:
        st.markdown(
            f"**Niveau de risque eleve.** {n_alerts:,} alertes ({alert_rate:.1%} du trafic) dont **{n_high} critiques**. "
            f"Ce volume d'alertes peut indiquer une attaque en cours ou un reseau compromis. "
            f"Action recommandee : examiner les sessions critiques en priorite."
        )

    # Confiance et accord
    extra_info = []
    if mean_conf is not None:
        if mean_conf >= 0.7:
            extra_info.append(f"Confiance moyenne : **{mean_conf:.0%}** — les predictions sont fiables.")
        elif mean_conf >= 0.5:
            extra_info.append(f"Confiance moyenne : **{mean_conf:.0%}** — certaines predictions meritent verification.")
        else:
            extra_info.append(
                f"Confiance moyenne : **{mean_conf:.0%}** — predictions peu fiables. "
                "Verifiez la qualite des donnees ou utilisez un format plus complet (PCAP)."
            )
    if model_agreement is not None:
        if model_agreement >= 0.95:
            extra_info.append(f"Accord RF/XGBoost : **{model_agreement:.1%}** — les modeles convergent.")
        else:
            n_disagree = int((1 - model_agreement) * n_total)
            extra_info.append(
                f"Accord RF/XGBoost : **{model_agreement:.1%}** — "
                f"**{n_disagree}** session(s) en desaccord, a investiguer."
            )

    if extra_info:
        for info in extra_info:
            st.markdown(f"- {info}")

    # Recommandations
    st.markdown("**Actions recommandees :**")
    actions = []
    if n_high > 0:
        actions.append(f"Investiguer les **{n_high} sessions critiques** (P > 0.8) dans *Analyse detaillee*")
    if n_alerts > 0:
        actions.append("Examiner la **distribution des probabilites** ci-dessous pour comprendre la repartition")
        actions.append("Utiliser la **Projection** pour visualiser les clusters de sessions suspectes")
    if n_alerts == 0:
        actions.append("Le trafic semble sain — verifiez periodiquement avec de nouvelles captures")
    if model_agreement is not None and model_agreement < 0.95:
        actions.append("Verifier les sessions ou RF et XGBoost sont en desaccord dans *Statistiques*")
    actions.append("Ajuster le **seuil** dans la sidebar si le taux de faux positifs/negatifs est trop eleve")

    for action in actions:
        st.markdown(f"- {action}")
