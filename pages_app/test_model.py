"""
Page Tester le modele — Selection de dataset OU import custom,
predictions, performance, drift detection, sessions cliquables.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import (explain, render_metric_card, render_quality_bar,
                               render_pedagogy, save_comparison_result)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Datasets disponibles ===
DATASETS = {
    "CIC-Darknet2020 — 5 000 sessions (distribution reelle)": {
        "file": "demo_sample.csv",
        "label": "CIC-Darknet2020 — 5 000 sessions (distribution reelle du dataset)",
        "features": 27,
        "source": "CIC-Darknet2020 (Univ. New Brunswick, 2020)",
        "attacks": "25 familles de malwares (trojans, ransomware, botnets)",
        "encrypted": True,
        "is_training_data": True,
    },
    "CIC-Darknet2020 — 1 000 sessions (test set, mix 50/50)": {
        "file": "external_test_sample.csv",
        "label": "CIC-Darknet2020 — 1 000 sessions (500 benignes + 500 malveillantes)",
        "features": 27,
        "source": "CIC-Darknet2020 (Univ. New Brunswick, 2020)",
        "attacks": "25 familles de malwares (trojans, ransomware, botnets)",
        "encrypted": True,
        "is_training_data": True,
    },
    "USTC-TFC2016 — Zeus, Tinba, Geodo, Miuref": {
        "file": "sample_ustc_tfc2016.csv",
        "label": "USTC-TFC2016 — 1 000 sessions (500 benignes + 500 malwares)",
        "features": 27,
        "source": "USTC-TFC2016 (Univ. Science & Technology of China, 2016)",
        "attacks": "Zeus (trojan), Tinba (banking), Geodo (botnet), Miuref (backdoor)",
        "encrypted": True,
        "is_training_data": False,
    },
    "USTC-TFC2016 — Cridex, Virut": {
        "file": "sample_ustc_cridex_virut.csv",
        "label": "USTC-TFC2016 — 1 000 sessions (500 benignes + 250 Cridex + 250 Virut)",
        "features": 27,
        "source": "USTC-TFC2016 (Univ. Science & Technology of China, 2016)",
        "attacks": "Cridex (trojan bancaire), Virut (virus polymorphe)",
        "encrypted": True,
        "is_training_data": False,
    },
    "USTC-TFC2016 — Htbot, Neris": {
        "file": "sample_ustc_htbot_neris.csv",
        "label": "USTC-TFC2016 — 1 000 sessions (500 benignes + 250 Htbot + 250 Neris)",
        "features": 27,
        "source": "USTC-TFC2016 (Univ. Science & Technology of China, 2016)",
        "attacks": "Htbot (botnet HTTP), Neris (botnet spam/DDoS)",
        "encrypted": True,
        "is_training_data": False,
    },
}

TRAINING_REF = {
    "accuracy": 99.50,
    "precision": 99.84,
    "recall": 99.16,
    "f1": 0.9950,
}


def render(models, session_features, config):

    st.header("Tester le modele")
    explain(
        "Evaluez les performances du Random Forest sur differents datasets de trafic chiffre, "
        "ou importez vos propres donnees (PCAP, CSV, Excel). Les resultats sont compares "
        "aux performances obtenues a l'entrainement (F1 = 0.995 sur CIC-Darknet2020)."
    )

    # === Deux onglets : Datasets inclus / Importer ===
    tab_datasets, tab_upload = st.tabs([
        "Datasets inclus",
        "Importer vos donnees"
    ])

    with tab_datasets:
        _render_dataset_selector(models, session_features, config)

    with tab_upload:
        _render_upload_section(models, session_features, config)


# =========================================================================
# TAB 1 : DATASETS INCLUS
# =========================================================================

def _render_dataset_selector(models, session_features, config):
    """Selecteur de datasets pre-configures."""

    selected = st.selectbox(
        "Choisir un dataset de test",
        list(DATASETS.keys()),
        key="dataset_selector",
    )
    ds = DATASETS[selected]

    # Charger si necessaire
    if st.session_state.get("_loaded_dataset") != ds["file"]:
        with st.spinner("Chargement et analyse en cours..."):
            _load_csv_dataset(ds["file"], ds["label"], ds["features"], session_features)
            st.session_state["_loaded_dataset"] = ds["file"]
            st.session_state["_current_ds_info"] = ds
        st.rerun()

    # Auto-load au premier affichage
    if "data" not in st.session_state or "probas" not in st.session_state:
        with st.spinner("Chargement du dataset par defaut..."):
            _load_csv_dataset(ds["file"], ds["label"], ds["features"], session_features)
            st.session_state["_loaded_dataset"] = ds["file"]
            st.session_state["_current_ds_info"] = ds
        st.rerun()

    st.caption(
        f"Source : {ds['source']} | Attaques : {ds['attacks']} | "
        f"Features : {ds['features']}/27"
    )

    # Seuil + qualite
    _render_threshold_and_quality(config, ds)

    # Pedagogie : contexte du dataset
    if not ds.get("is_training_data"):
        render_pedagogy(
            "<strong>Dataset externe</strong> — Ce dataset n'a pas ete utilise pour l'entrainement. "
            "Les performances peuvent etre differentes car les distributions de features "
            "et les familles de malwares sont differentes. "
            "C'est un <strong>test de generalisation</strong> : on evalue si le modele detecte "
            "des menaces qu'il n'a jamais vues."
        )

    # Resultats
    _render_full_results(models, session_features, config, ds)


# =========================================================================
# TAB 2 : IMPORT CUSTOM
# =========================================================================

def _render_upload_section(models, session_features, config):
    """Section d'import de fichiers custom."""

    st.markdown("""
    **Formats supportes :**

    | Format | Type | Qualite attendue |
    |--------|------|-----------------|
    | **PCAP / PCAPNG** | Capture brute | Optimal (27/27 features) |
    | **CSV CIC-Darknet2020** | Session-level | Optimal |
    | **CSV CICFlowMeter** | Session-level | Bon (~20/27) |
    | **CSV Wireshark** | Paquet-level (agregation auto) | Variable |
    | **CSV tshark (etendu)** | Paquet-level | Optimal |
    | **Excel (.xlsx)** | Auto-detecte | Variable |
    | **Zeek conn.log** | Session-level | Partiel (~5/27) |
    """)

    uploaded = st.file_uploader(
        "Importez votre fichier",
        type=["pcap", "pcapng", "csv", "xlsx", "xls", "log", "tsv"],
        help="Taille max : 500 Mo. Les fichiers PCAP sont analyses directement.",
    )

    if uploaded is not None:
        with st.spinner(f"Analyse de {uploaded.name} en cours..."):
            if _process_upload(uploaded, session_features):
                st.session_state["_loaded_dataset"] = "__custom__"
                st.session_state["_current_ds_info"] = {
                    "file": "__custom__",
                    "source": uploaded.name,
                    "attacks": "Inconnu",
                    "features": st.session_state.get("feature_quality", {}).get("available", 27),
                    "is_training_data": False,
                }
                st.rerun()

    # Si des donnees custom sont chargees, afficher les resultats
    if st.session_state.get("_loaded_dataset") == "__custom__" and "probas" in st.session_state:
        ds_info = st.session_state.get("_current_ds_info", {})
        _render_threshold_and_quality(config, ds_info)
        _render_full_results(models, session_features, config, ds_info)


# =========================================================================
# COMPOSANTS PARTAGES
# =========================================================================

def _render_threshold_and_quality(config, ds_info):
    """Seuil de detection + barre de qualite."""
    col_t, col_q = st.columns([1, 1])

    with col_t:
        threshold = st.slider(
            "Seuil de detection",
            min_value=0.1, max_value=0.9,
            value=config["threshold"], step=0.05,
            help="Probabilite au-dessus de laquelle une session est classee comme malveillante.",
            key=f"threshold_slider_{ds_info.get('file', 'custom')}",
        )
        config["threshold"] = threshold

    with col_q:
        n_feat = ds_info.get("features", 27)
        render_quality_bar(n_feat, 27)


def _render_full_results(models, session_features, config, ds_info):
    """Pipeline complet d'affichage des resultats."""
    if "data" not in st.session_state or "probas" not in st.session_state:
        return

    df = st.session_state["data"]
    probas = st.session_state["probas"]
    threshold = config["threshold"]
    preds = (probas >= threshold).astype(int)
    has_labels = "label" in df.columns

    if has_labels:
        y_true = df["label"].values.astype(int)

    n_total = len(df)
    n_alerts = int(preds.sum())

    # === Diagnostic de compatibilite (drift detection) ===
    _render_drift_diagnostic(session_features, ds_info)

    # === Encart differences par rapport a l'entrainement ===
    _render_training_diff(ds_info, config)

    # === Composition du dataset (si labels) ===
    if has_labels:
        n_real_mal = int(y_true.sum())
        n_real_ben = n_total - n_real_mal
        st.markdown("---")
        st.subheader("Composition du dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Sessions totales", f"{n_total:,}", "blue")
        with col2:
            render_metric_card("Benignes (reel)", f"{n_real_ben:,}", "green")
        with col3:
            render_metric_card("Malveillantes (reel)", f"{n_real_mal:,}", "red")

    # === Predictions du modele ===
    st.markdown("---")
    st.subheader("Predictions du modele")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Predites benignes", f"{n_total - n_alerts:,}", "green")
    with col2:
        render_metric_card("Predites malveillantes", f"{n_alerts:,}", "red" if n_alerts > 0 else "green")
    with col3:
        render_metric_card("Seuil", f"{threshold}", "blue")

    # === Performance (si labels) ===
    if has_labels:
        _render_performance(preds, probas, y_true, ds_info, config)

    # === Distribution des probabilites ===
    st.markdown("---")
    st.subheader("Distribution des probabilites")

    render_pedagogy(
        "<strong>Comment lire ce graphique :</strong> Un bon modele produit une distribution "
        "<strong>bimodale</strong> — un pic pres de 0 (sessions benignes, le modele est sur) "
        "et un pic pres de 1 (sessions malveillantes, le modele est sur). "
        "Les sessions entre 0.3 et 0.7 sont dans une <strong>zone d'incertitude</strong> "
        "— le modele hesite. La ligne rouge represente le seuil de decision."
    )

    fig = go.Figure()
    if has_labels:
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
        fig.add_trace(go.Histogram(
            x=probas, nbinsx=50, name="Toutes", marker_color="#3b82f6", opacity=0.7
        ))

    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil = {threshold}")
    # Zone d'incertitude
    fig.add_vrect(x0=0.3, x1=0.7, fillcolor="rgba(245,158,11,0.08)",
                  line_width=0, annotation_text="Zone d'incertitude",
                  annotation_position="top left",
                  annotation_font_color="#f59e0b")
    fig.update_layout(
        xaxis_title="Probabilite (0 = benin, 1 = malveillant)",
        yaxis_title="Nombre de sessions",
        template="plotly_dark", height=350, margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Top sessions suspectes (cliquables) ===
    _render_suspect_sessions(df, probas, preds, has_labels, y_true if has_labels else None, session_features)

    # === Export ===
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        try:
            from src.report import generate_csv_report
            csv_data = generate_csv_report(
                df, probas, preds, session_features,
                confidence=st.session_state.get("confidence"),
            )
            st.download_button("Telecharger CSV", data=csv_data,
                               file_name="resultats.csv", mime="text/csv",
                               use_container_width=True)
        except Exception:
            pass
    with col_exp2:
        try:
            from src.report import generate_pdf_report
            pdf_data = generate_pdf_report(
                df, probas, preds, config, session_features,
                y_true=st.session_state.get("y_true"),
                confidence=st.session_state.get("confidence"),
                probas_xgb=st.session_state.get("probas_xgb"),
                feature_quality=st.session_state.get("feature_quality"),
            )
            st.download_button("Telecharger PDF", data=pdf_data,
                               file_name="rapport.pdf", mime="application/pdf",
                               use_container_width=True)
        except Exception:
            pass

    # === Sauvegarder pour comparaison ===
    if has_labels:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        save_comparison_result(
            ds_info.get("label", ds_info.get("source", "Inconnu")),
            {
                "accuracy": 100 * accuracy_score(y_true, preds),
                "precision": 100 * precision_score(y_true, preds, zero_division=0),
                "recall": 100 * recall_score(y_true, preds, zero_division=0),
                "f1": f1_score(y_true, preds, zero_division=0),
                "n_sessions": n_total,
                "n_alerts": n_alerts,
                "threshold": threshold,
                "features": ds_info.get("features", 27),
                "source": ds_info.get("source", ""),
            }
        )


# =========================================================================
# DRIFT DETECTION
# =========================================================================

def _render_drift_diagnostic(session_features, ds_info):
    """Diagnostic de compatibilite : compare les features importees aux stats d'entrainement."""
    if "X" not in st.session_state:
        return

    X = st.session_state["X"]
    if X.shape[0] < 5:
        return

    from src.models import load_training_stats
    training_stats = load_training_stats(tuple(session_features))
    if training_stats is None:
        return

    from src.feature_drift import compute_feature_drift, transferability_label
    drift = compute_feature_drift(X, session_features, training_stats)
    if drift is None:
        return

    st.session_state["_drift_result"] = drift

    st.markdown("---")
    st.subheader("Diagnostic de compatibilite")

    label, color = transferability_label(drift["global_score"])

    col_score, col_detail = st.columns([1, 2])
    with col_score:
        render_metric_card("Score de compatibilite", f"{drift['global_score']:.0f}/100", color)
        st.caption(label)

    with col_detail:
        # Radar par categorie
        cats = list(drift["category_scores"].keys())
        vals = [drift["category_scores"][c] for c in cats]
        cats_closed = cats + [cats[0]]
        vals_closed = vals + [vals[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name="Compatibilite",
            fillcolor=f"rgba({'16,185,129' if drift['global_score'] >= 60 else '245,158,11'}, 0.2)",
            line=dict(color="#10b981" if drift["global_score"] >= 60 else "#f59e0b", width=2)
        ))
        fig.add_trace(go.Scatterpolar(
            r=[100] * len(cats_closed), theta=cats_closed,
            fill="toself", name="Reference (entrainement)",
            fillcolor="rgba(59,130,246,0.05)",
            line=dict(color="#3b82f6", width=1, dash="dash")
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                               gridcolor="#334155", color="#94a3b8"),
                angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)"
            ),
            template="plotly_dark", height=300,
            margin=dict(t=20, b=20, l=40, r=40),
            showlegend=True,
            legend=dict(y=-0.15, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Features en derive
    if drift["drifted_features"]:
        n_drift = len(drift["drifted_features"])
        with st.expander(f"{n_drift} feature(s) en derive par rapport a l'entrainement"):
            for feat in drift["drifted_features"][:10]:
                drift_level = "drift-bad" if feat["drift_score"] > 3 else ("drift-warn" if feat["drift_score"] > 1.5 else "drift-ok")
                st.markdown(
                    f'**{feat["name"]}** '
                    f'<span class="drift-indicator {drift_level}">drift = {feat["drift_score"]:.1f}</span>'
                    f' — moyenne entrainement: {feat["train_mean"]:.3f}, '
                    f'moyenne importee: {feat["import_mean"]:.3f}',
                    unsafe_allow_html=True
                )

    render_pedagogy(
        f"<strong>Score de compatibilite : {drift['global_score']:.0f}/100</strong> — "
        "Ce score mesure a quel point les distributions de vos donnees correspondent "
        "a celles vues par le modele pendant l'entrainement. "
        "Un score &lt; 60 signifie que les predictions seront moins fiables. "
        "Ce n'est pas un defaut du modele : c'est une limite inherente au Machine Learning "
        "supervise, qui performe mieux sur des donnees similaires a celles d'entrainement."
    )


# =========================================================================
# DIFFERENCES PAR RAPPORT A L'ENTRAINEMENT
# =========================================================================

def _render_training_diff(ds_info, config):
    """Affiche un encart si les conditions different de l'entrainement."""
    diffs = []
    is_training_data = ds_info.get("is_training_data", False)

    n_feat = ds_info.get("features", 27)
    if n_feat < 27:
        diffs.append(f"**Features** : {n_feat}/27 disponibles ({27 - n_feat} features manquantes mises a zero)")

    if not is_training_data:
        diffs.append("**Source** : dataset externe (entrainement = CIC-Darknet2020)")

    threshold = config.get("threshold", 0.5)
    if threshold != 0.5:
        diffs.append(f"**Seuil** : {threshold} (entrainement = 0.5)")

    if diffs:
        diff_text = "<br>".join([f"- {d}" for d in diffs])
        st.markdown(
            f'<div class="metric-card"><h3>Differences par rapport a l\'entrainement</h3>'
            f'<div style="text-align:left;font-size:0.9em;">{diff_text}</div></div>',
            unsafe_allow_html=True,
        )


# =========================================================================
# SESSIONS SUSPECTES (CLIQUABLES)
# =========================================================================

def _render_suspect_sessions(df, probas, preds, has_labels, y_true, session_features):
    """Tableau des sessions suspectes avec detail cliquable."""
    st.markdown("---")
    st.subheader("Sessions les plus suspectes")

    render_pedagogy(
        "<strong>Comment explorer les resultats :</strong> Le tableau ci-dessous liste les sessions "
        "triees par probabilite de malveillance. Cliquez sur <strong>Voir le detail</strong> "
        "pour une session specifique pour comprendre <em>pourquoi</em> le modele l'a classee ainsi. "
        "Vous pouvez aussi aller sur la page <strong>Detail d'une session</strong> dans le menu "
        "pour une analyse complete (radar de risque, SHAP, analyse comportementale)."
    )

    df_display = df.copy()
    df_display["probabilite"] = probas
    df_display["verdict"] = np.where(preds == 1, "SUSPECT", "Benin")
    if has_labels:
        df_display["verite_terrain"] = np.where(y_true == 1, "Malveillant", "Benin")
        df_display["correct"] = np.where(
            preds == y_true, "Oui", "Non"
        )

    display_cols = ["probabilite", "verdict"]
    if has_labels:
        display_cols.extend(["verite_terrain", "correct"])

    df_sorted = df_display[display_cols].sort_values("probabilite", ascending=False)

    # Afficher le tableau
    styled = df_sorted.head(100).style.background_gradient(
        subset=["probabilite"], cmap="RdYlGn_r", vmin=0, vmax=1
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # Detail inline d'une session
    st.markdown("---")
    st.subheader("Examiner une session")

    top_indices = np.argsort(probas)[::-1][:50]
    selected_idx = st.selectbox(
        "Choisir une session a examiner",
        top_indices,
        format_func=lambda i: (
            f"Session {i} — P={probas[i]:.4f} "
            f"({'SUSPECT' if preds[i] == 1 else 'Benin'})"
            + (f" | Reel: {'Malveillant' if y_true[i] == 1 else 'Benin'}" if has_labels else "")
        ),
        key="session_detail_selector"
    )

    if selected_idx is not None:
        _render_session_quick_detail(selected_idx, probas, preds, session_features, has_labels, y_true)


def _render_session_quick_detail(idx, probas, preds, session_features, has_labels, y_true):
    """Affiche un apercu rapide de la session selectionnee."""
    X = st.session_state.get("X")
    if X is None:
        return

    proba = probas[idx]
    verdict = "SUSPECT" if preds[idx] == 1 else "Benin"

    st.markdown('<div class="session-detail-box">', unsafe_allow_html=True)

    # Verdict
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "red" if proba >= 0.5 else ("yellow" if proba >= 0.3 else "green")
        render_metric_card("Probabilite", f"{proba:.4f}", color)
    with col2:
        render_metric_card("Verdict", verdict, "red" if verdict == "SUSPECT" else "green")
    with col3:
        level = "Critique" if proba >= 0.8 else ("Eleve" if proba >= 0.5 else ("Moyen" if proba >= 0.3 else "Faible"))
        level_color = "red" if proba >= 0.8 else ("red" if proba >= 0.5 else ("yellow" if proba >= 0.3 else "green"))
        render_metric_card("Niveau de risque", level, level_color)
    with col4:
        if has_labels and y_true is not None:
            label = "Malveillant" if y_true[idx] == 1 else "Benin"
            correct = (y_true[idx] == preds[idx])
            render_metric_card("Verite terrain", label, "green" if correct else "red")
        else:
            confidence = st.session_state.get("confidence")
            if confidence is not None:
                from src.confidence import confidence_label
                conf_text, conf_color = confidence_label(confidence[idx])
                render_metric_card("Confiance", f"{confidence[idx]:.0%}", conf_color)
            else:
                render_metric_card("Verite terrain", "N/A", "blue")

    # Profil de risque rapide (radar simplifie)
    feature_vals = X[idx]
    from src.models import load_training_stats
    training_stats = load_training_stats(tuple(session_features))

    if training_stats is not None:
        mean_ref = training_stats["mean"]
        std_ref = training_stats["std"]
    else:
        mean_ref = X.mean(axis=0)
        std_ref = X.std(axis=0)

    z_scores = np.where(std_ref > 0, (feature_vals - mean_ref) / std_ref, 0)

    col_radar, col_anomalies = st.columns([1, 1])

    with col_radar:
        st.markdown("**Profil de risque**")
        categories = {
            "Timing": ["max_Interval_of_arrival_time_of_backward_traffic_enc",
                        "max_Interval_of_arrival_time_of_backward_traffic_ratio",
                        "flow_duration_of_backward_traffic_ratio"],
            "Volume": ["min_forward_packet_length", "min_backward_packet_length",
                        "std_forward_packet_length_enc", "std_forward_packet_length_ratio"],
            "IP/Ratio": ["IPratio_enc", "IPratio_ratio", "max_length_of_IP_packet_ratio"],
            "TCP": ["max_TCP_windows_size_value_forward_traffic_ratio",
                    "total_TCP_windows_size_value_forward_traffic_ratio",
                    "max_Change_values_of_TCP_windows_length_per_session"],
            "TTL": ["min_ttl_forward_traffic", "max_ttl_backward_traffic_enc",
                    "median_ttl_backward_traffic"],
        }

        cats = []
        vals = []
        for cat, feats in categories.items():
            indices = [session_features.index(f) for f in feats if f in session_features]
            if indices:
                score = np.mean(np.abs(z_scores[indices]))
                cats.append(cat)
                vals.append(min(float(score), 5))

        if cats:
            cats_closed = cats + [cats[0]]
            vals_closed = vals + [vals[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.3)" if proba >= 0.5 else "rgba(59, 130, 246, 0.3)",
                line=dict(color="#ef4444" if proba >= 0.5 else "#3b82f6", width=2)
            ))
            fig.add_trace(go.Scatterpolar(
                r=[1] * len(cats_closed), theta=cats_closed,
                fill="toself",
                fillcolor="rgba(59,130,246,0.05)",
                line=dict(color="#3b82f6", width=1, dash="dash")
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5], gridcolor="#334155"),
                    angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
                    bgcolor="rgba(0,0,0,0)"
                ),
                template="plotly_dark", height=280,
                margin=dict(t=20, b=20, l=30, r=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_anomalies:
        st.markdown("**Features les plus anormales**")
        sorted_idx = np.argsort(np.abs(z_scores))[::-1][:5]
        for i in sorted_idx:
            z = z_scores[i]
            if abs(z) < 0.5:
                continue
            name = session_features[i].replace("_", " ")[:50]
            color = "#ef4444" if z > 2 else ("#3b82f6" if z < -2 else "#94a3b8")
            direction = "eleve" if z > 0 else "bas"
            st.markdown(
                f'<span style="color:{color};font-weight:600">Z={z:+.1f}</span> '
                f'**{name}** — anormalement {direction}',
                unsafe_allow_html=True
            )

    # Analyse comportementale rapide
    from src.temporal import analyze_session_timing
    temporal = analyze_session_timing(feature_vals, session_features, mean_ref, std_ref)
    if temporal.get("indicators"):
        st.markdown("**Indicateurs comportementaux :**")
        for ind in temporal["indicators"][:3]:
            severity_color = "#ef4444" if ind["severity"] == "high" else "#f59e0b"
            st.markdown(
                f'<span style="color:{severity_color};">&#9679;</span> '
                f'{ind["description"][:150]}',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Pour une analyse complete (SHAP, sessions similaires, feedback), "
        "rendez-vous sur la page **Detail d'une session** dans le menu."
    )


# =========================================================================
# PERFORMANCE
# =========================================================================

def _render_performance(preds, probas, y_true, ds_info, config):
    """Performance test vs entrainement + matrice de confusion + explication pedagogique."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)

    acc = 100 * accuracy_score(y_true, preds)
    prec = 100 * precision_score(y_true, preds, zero_division=0)
    rec = 100 * recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    ref = TRAINING_REF

    st.markdown("---")
    st.subheader("Performance du modele")

    # Tableau comparatif
    def _delta(test_val, ref_val):
        diff = test_val - ref_val
        if abs(diff) < 0.01:
            return "="
        return f"{diff:+.2f}%"

    comparison = {
        "Metrique": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Ce dataset": [f"{acc:.2f}%", f"{prec:.2f}%", f"{rec:.2f}%", f"{f1:.4f}"],
        "Entrainement (ref)": [
            f"{ref['accuracy']:.2f}%", f"{ref['precision']:.2f}%",
            f"{ref['recall']:.2f}%", f"{ref['f1']:.4f}",
        ],
        "Ecart": [
            _delta(acc, ref["accuracy"]), _delta(prec, ref["precision"]),
            _delta(rec, ref["recall"]), _delta(f1 * 100, ref["f1"] * 100),
        ],
    }
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Accuracy", f"{acc:.2f}%", "blue")
    with col2:
        render_metric_card("Precision", f"{prec:.2f}%", "green")
    with col3:
        render_metric_card("Recall", f"{rec:.2f}%", "yellow")
    with col4:
        f1_color = "green" if f1 > 0.9 else ("yellow" if f1 > 0.5 else "red")
        render_metric_card("F1-score", f"{f1:.4f}", f1_color)

    # Matrice de confusion
    col_cm, col_desc = st.columns(2)
    with col_cm:
        z = [[tn, fp], [fn, tp]]
        text = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
                [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]
        fig = go.Figure(data=go.Heatmap(
            z=z, x=["Predit Benin", "Predit Malveillant"],
            y=["Reel Benin", "Reel Malveillant"],
            text=text, texttemplate="%{text}", textfont=dict(size=14),
            colorscale=[[0, "#1e293b"], [1, "#3b82f6"]], showscale=False
        ))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col_desc:
        render_pedagogy(
            "<strong>Lire la matrice de confusion :</strong><br>"
            f"<strong>TP = {tp:,}</strong> — malwares correctement detectes (vrais positifs).<br>"
            f"<strong>TN = {tn:,}</strong> — sessions benignes correctement ignorees.<br>"
            f"<strong>FP = {fp:,}</strong> — fausses alertes (le modele a tort, la session est benigne).<br>"
            f"<strong>FN = {fn:,}</strong> — malwares rates (le plus dangereux : le modele n'a pas vu la menace).<br><br>"
            "L'objectif est de maximiser TP et TN tout en minimisant FN."
        )

    # === Explication pedagogique si performance faible ===
    if f1 < 0.9:
        st.markdown("---")
        st.subheader("Pourquoi les performances sont differentes ?")

        if f1 < 0.5:
            st.error(f"**F1 = {f1:.4f}** — Le modele ne detecte pas les menaces de ce dataset.")
        else:
            st.warning(f"**F1 = {f1:.4f}** — Performance degradee par rapport a l'entrainement (F1 = 0.9950).")

        render_pedagogy(
            "C'est un resultat <strong>attendu et normal</strong> en Machine Learning. "
            "Voici les raisons possibles :"
        )

        reasons = []
        ds_file = ds_info.get("file", "")
        n_feat = ds_info.get("features", 27)

        # Raison 1 : dataset externe
        if not ds_info.get("is_training_data", False):
            reasons.append(
                "**Donnees differentes de l'entrainement** — Le modele a appris les patterns "
                "statistiques du dataset CIC-Darknet2020. Un dataset extrait par un processus "
                "different produit des distributions de features differentes, meme si les noms "
                "de colonnes sont identiques."
            )

        # Raison 2 : familles de malware
        if "ustc" in ds_file:
            reasons.append(
                f"**Familles de malware inconnues** — Ce dataset contient "
                f"*{ds_info.get('attacks', 'inconnu')}*. Ces malwares datent de 2016 "
                "et ont des patterns reseau differents des malwares darknet de 2020 "
                "sur lesquels le modele a ete entraine."
            )

        # Raison 3 : features manquantes
        if n_feat < 27:
            reasons.append(
                f"**Features incompletes** — Seules {n_feat}/27 features sont disponibles. "
                "Les features manquantes sont mises a zero, ce qui fausse les predictions."
            )

        # Raison generale
        reasons.append(
            "**Limite inherente au ML supervise** — Un modele supervise est specialise "
            "sur ses donnees d'entrainement. Il ne generalise pas automatiquement a des "
            "donnees differentes. Pour ameliorer la generalisation, il faudrait re-entrainer "
            "le modele sur un corpus plus diversifie."
        )

        # Drift
        drift = st.session_state.get("_drift_result")
        if drift and drift["global_score"] < 60:
            reasons.insert(0,
                f"**Derive des features detectee** (score = {drift['global_score']:.0f}/100) — "
                f"Les distributions de {len(drift['drifted_features'])} features s'ecartent "
                "significativement de celles vues a l'entrainement. Le modele recoit des "
                "valeurs qu'il n'a jamais rencontrees."
            )

        for i, r in enumerate(reasons, 1):
            st.markdown(f"{i}. {r}")


# =========================================================================
# CHARGEMENT
# =========================================================================

def _load_csv_dataset(filename, source_label, n_features, session_features):
    """Charge un CSV pre-adapte depuis data/."""
    path = os.path.join(APP_DIR, "data", filename)
    if not os.path.exists(path):
        st.error(f"Fichier introuvable : {filename}")
        return False

    df = pd.read_csv(path, low_memory=False)
    st.session_state["data"] = df
    st.session_state["data_source"] = source_label
    st.session_state["feature_quality"] = {"total": 27, "available": n_features}
    return _run_predictions(session_features)


def _process_upload(uploaded, session_features):
    """Traite un fichier uploade (multi-format)."""
    filename = uploaded.name.lower()

    if uploaded.size > 500 * 1024 * 1024:
        st.error("Fichier trop volumineux (> 500 Mo).")
        return False
    if uploaded.size == 0:
        st.error("Le fichier est vide.")
        return False

    if filename.endswith(".pcap") or filename.endswith(".pcapng"):
        try:
            from src.feature_extraction import extract_sessions_from_pcap, compute_session_features
            pcap_bytes = uploaded.read()
            sessions_dict = extract_sessions_from_pcap(pcap_bytes)
            if not sessions_dict:
                st.error("Aucune session TCP/UDP trouvee.")
                return False
            df = compute_session_features(sessions_dict, session_features)
            st.session_state["data"] = df
            st.session_state["data_source"] = f"PCAP : {uploaded.name} ({len(sessions_dict)} sessions)"
            st.session_state["feature_quality"] = {"total": 27, "available": 27}
        except Exception as e:
            st.error(f"Erreur PCAP : {e}")
            return False

    elif filename.endswith(".log") or filename.endswith(".tsv"):
        try:
            content = uploaded.read().decode("utf-8", errors="replace")
            from src.feature_extraction import parse_zeek_connlog, adapt_zeek_connlog
            df_raw = parse_zeek_connlog(content)
            if df_raw is not None and "id.orig_h" in df_raw.columns:
                df = adapt_zeek_connlog(df_raw, session_features)
                n = sum(1 for f in session_features if f in df.columns and (df[f] != 0).any())
                st.session_state["data"] = df
                st.session_state["data_source"] = f"Zeek : {uploaded.name} ({len(df)} sessions)"
                st.session_state["feature_quality"] = {"total": 27, "available": n}
            else:
                uploaded.seek(0)
                df_raw = pd.read_csv(uploaded, sep="\t", low_memory=False, comment="#")
                return _process_tabular(df_raw, uploaded.name, session_features)
        except Exception as e:
            st.error(f"Erreur Zeek/TSV : {e}")
            return False

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        try:
            df_raw = pd.read_excel(uploaded, engine="openpyxl" if filename.endswith(".xlsx") else None)
            return _process_tabular(df_raw, uploaded.name, session_features)
        except Exception as e:
            st.error(f"Erreur Excel : {e}")
            return False

    else:
        try:
            df_raw = pd.read_csv(uploaded, low_memory=False)
            return _process_tabular(df_raw, uploaded.name, session_features)
        except Exception as e:
            st.error(f"Erreur CSV : {e}")
            return False

    return _run_predictions(session_features)


def _process_tabular(df_raw, filename, session_features):
    """Detection de format + adaptation."""
    if df_raw.empty or len(df_raw.columns) < 2:
        st.error("Fichier vide ou invalide.")
        return False

    from src.feature_extraction import detect_dataset_format, adapt_dataframe, adapt_cicflowmeter

    fmt, matched, missing, fmt_type = detect_dataset_format(df_raw, session_features)

    if fmt == "CIC-Darknet2020":
        df = adapt_dataframe(df_raw, session_features) if missing else df_raw
        st.session_state["data"] = df
        st.session_state["data_source"] = f"CIC-Darknet2020 : {filename} ({len(df)} sessions)"
        st.session_state["feature_quality"] = {"total": 27, "available": len(matched)}

    elif fmt == "CICFlowMeter":
        df = adapt_cicflowmeter(df_raw, session_features)
        n = sum(1 for f in session_features if f in df.columns and (df[f] != 0).any())
        st.session_state["data"] = df
        st.session_state["data_source"] = f"CICFlowMeter : {filename} ({len(df)} sessions)"
        st.session_state["feature_quality"] = {"total": 27, "available": n}

    elif fmt_type == "packet_level":
        from src.csv_aggregation import detect_packet_csv, aggregate_packets_to_sessions
        is_packet, col_map, _ = detect_packet_csv(df_raw)
        if not is_packet:
            st.error("Format non reconnu.")
            return False
        df, quality = aggregate_packets_to_sessions(df_raw, col_map, session_features)
        if df.empty:
            st.error("Aucune session trouvee apres aggregation.")
            return False
        st.session_state["data"] = df
        st.session_state["data_source"] = f"Wireshark : {filename} ({quality['sessions_count']} sessions)"
        st.session_state["feature_quality"] = quality

    elif fmt in ("Compatible (partiel)", "Partiellement compatible"):
        df = adapt_dataframe(df_raw, session_features)
        st.session_state["data"] = df
        st.session_state["data_source"] = f"{filename} ({len(df)} sessions, {len(matched)}/27 features)"
        st.session_state["feature_quality"] = {"total": 27, "available": len(matched)}

    else:
        st.error(
            "**Format non reconnu.** Formats supportes : "
            "PCAP, CSV CIC-Darknet2020, CSV CICFlowMeter, CSV Wireshark, Excel, Zeek conn.log."
        )
        return False

    return _run_predictions(session_features)


def _run_predictions(session_features):
    """Lance RF + XGBoost + confiance."""
    from src.models import load_models
    models, _ = load_models()

    if "rf_session" not in models or "data" not in st.session_state:
        return True

    df = st.session_state["data"]
    avail = [f for f in session_features if f in df.columns]
    if len(avail) != len(session_features):
        st.error(f"Features manquantes ({len(avail)}/{len(session_features)}).")
        return False

    X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)
    probas_rf = models["rf_session"].predict_proba(X)[:, 1]
    preds = (probas_rf >= 0.5).astype(int)

    st.session_state["probas"] = probas_rf
    st.session_state["preds"] = preds
    st.session_state["preds_rf"] = preds
    st.session_state["X"] = X
    st.session_state["if_preds"] = None
    st.session_state["if_scores"] = None

    probas_xgb = None
    if "xgboost" in models:
        try:
            probas_xgb = models["xgboost"].predict_proba(X)[:, 1]
            st.session_state["probas_xgb"] = probas_xgb
        except Exception:
            pass

    from src.confidence import compute_confidence_scores
    fq = st.session_state.get("feature_quality")
    st.session_state["confidence"] = compute_confidence_scores(
        probas_rf, probas_xgb=probas_xgb, feature_quality=fq
    )

    if "label" in df.columns:
        st.session_state["y_true"] = df["label"].values.astype(int)

    fq_avail = fq.get("available", 27) if fq else 27
    st.session_state["recommended_threshold"] = 0.5 if fq_avail >= 24 else (0.4 if fq_avail >= 18 else 0.35)

    return True
