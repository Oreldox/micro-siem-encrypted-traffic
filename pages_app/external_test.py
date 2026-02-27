"""
Page Test Externe — Evaluer la generalisation du modele sur un dataset jamais vu
pendant l'entrainement. Comparaison avec les performances de reference.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# PERFORMANCES DE REFERENCE (dataset d'entrainement / test original)
# ============================================================================

TRAINING_PERF = {
    "accuracy": 99.50,
    "precision": 99.84,
    "recall": 99.16,
    "f1": 0.9950,
    "fn": 511,
    "fp": 98,
    "total_test": 122132,
}


def render(models, session_features, config):
    """Point d'entree de la page Test Externe."""

    # ------------------------------------------------------------------
    # Phase 1 : landing — aucune donnee externe chargee
    # ------------------------------------------------------------------
    if "ext_probas" not in st.session_state:
        _render_landing(models, session_features, config)
        return

    # ------------------------------------------------------------------
    # Phase 3 : affichage des resultats
    # ------------------------------------------------------------------
    _render_results(models, session_features, config)


# =========================================================================
# PHASE 1 — LANDING
# =========================================================================

def _render_landing(models, session_features, config):
    """Affiche la page d'accueil avec les deux options de chargement."""

    st.markdown("""
    <div class="hero-banner">
        <h1>Test sur dataset externe</h1>
        <p>Evaluez la capacite de generalisation du modele en le testant sur des donnees
        qu'il n'a <strong>jamais vues</strong> pendant l'entrainement.</p>
    </div>
    """, unsafe_allow_html=True)

    explain(
        "Cette page permet de verifier si le modele fonctionne bien sur de nouvelles donnees. "
        "Un bon modele doit conserver des performances proches de celles obtenues sur le dataset "
        "d'entrainement. Une chute importante indiquerait un <strong>surapprentissage</strong> (overfitting)."
    )

    st.markdown("### Choisissez une source de donnees")

    col_included, col_upload = st.columns(2)

    # --- Option A : dataset inclus ---
    with col_included:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset de test inclus</h3>
            <div class="value blue" style="font-size:1.2rem">1 000 sessions CIC-Darknet2020</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Echantillon aleatoire du jeu de test original, jamais utilise pour "
            "l'entrainement. Contient des labels pour la comparaison de metriques."
        )
        if st.button("Charger le dataset inclus", type="primary", use_container_width=True,
                      key="ext_load_included"):
            _load_included_dataset(models, session_features, config)

    # --- Option B : upload ---
    with col_upload:
        st.markdown("""
        <div class="metric-card">
            <h3>Votre propre dataset</h3>
            <div class="value green" style="font-size:1.2rem">CSV / PCAP</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Importez un fichier CSV (CIC-Darknet2020, CICFlowMeter, Wireshark) "
            "ou PCAP/PCAPNG pour tester le modele sur vos donnees."
        )
        uploaded = st.file_uploader(
            "Importer un fichier",
            type=["csv", "pcap", "pcapng", "xlsx", "xls", "log", "tsv"],
            label_visibility="collapsed",
            key="ext_file_uploader",
            help="Formats : CSV (CIC-Darknet2020, CICFlowMeter, Wireshark), PCAP/PCAPNG, Excel, Zeek",
        )
        if uploaded is not None:
            _load_uploaded_dataset(uploaded, models, session_features, config)


# =========================================================================
# PHASE 2 — CHARGEMENT DES DONNEES
# =========================================================================

def _load_included_dataset(models, session_features, config):
    """Charge le dataset de test inclus et lance les predictions."""
    sample_path = os.path.join(APP_DIR, "data", "external_test_sample.csv")

    if not os.path.exists(sample_path):
        st.error("Fichier external_test_sample.csv introuvable dans le dossier data/.")
        return

    df = pd.read_csv(sample_path, low_memory=False)
    st.session_state["ext_data"] = df
    st.session_state["ext_source"] = "Dataset de test inclus (1 000 sessions)"

    _run_ext_predictions(df, models, session_features, config)
    st.rerun()


def _load_uploaded_dataset(uploaded, models, session_features, config):
    """Charge un fichier uploade, detecte le format, adapte les features."""
    filename = uploaded.name.lower()

    try:
        # --- PCAP / PCAPNG ---
        if filename.endswith(".pcap") or filename.endswith(".pcapng"):
            from src.feature_extraction import extract_sessions_from_pcap, compute_session_features
            pcap_bytes = uploaded.read()
            sessions_dict = extract_sessions_from_pcap(pcap_bytes)
            if not sessions_dict:
                st.error("Aucune session TCP/UDP trouvee dans le fichier PCAP.")
                return
            df = compute_session_features(sessions_dict, session_features)
            st.session_state["ext_data"] = df
            st.session_state["ext_source"] = f"PCAP : {uploaded.name} ({len(sessions_dict)} sessions)"

        # --- ZEEK ---
        elif filename.endswith(".log") or filename.endswith(".tsv"):
            content = uploaded.read().decode("utf-8", errors="replace")
            from src.feature_extraction import parse_zeek_connlog, adapt_zeek_connlog
            df_raw = parse_zeek_connlog(content)
            if df_raw is not None and "id.orig_h" in df_raw.columns:
                df = adapt_zeek_connlog(df_raw, session_features)
            else:
                uploaded.seek(0)
                df_raw = pd.read_csv(uploaded, sep="\t", low_memory=False, comment="#")
                df = _adapt_tabular(df_raw, session_features)
                if df is None:
                    return
            st.session_state["ext_data"] = df
            st.session_state["ext_source"] = f"Zeek/TSV : {uploaded.name} ({len(df)} sessions)"

        # --- EXCEL ---
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df_raw = pd.read_excel(uploaded, engine="openpyxl" if filename.endswith(".xlsx") else None)
            df = _adapt_tabular(df_raw, session_features)
            if df is None:
                return
            st.session_state["ext_data"] = df
            st.session_state["ext_source"] = f"Excel : {uploaded.name} ({len(df)} sessions)"

        # --- CSV ---
        else:
            df_raw = pd.read_csv(uploaded, low_memory=False)
            df = _adapt_tabular(df_raw, session_features)
            if df is None:
                return
            st.session_state["ext_data"] = df
            st.session_state["ext_source"] = f"CSV : {uploaded.name} ({len(df)} sessions)"

        df = st.session_state["ext_data"]
        _run_ext_predictions(df, models, session_features, config)
        st.rerun()

    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")


def _adapt_tabular(df_raw, session_features):
    """Detecte le format d'un DataFrame tabulaire et l'adapte aux features du modele."""
    from src.feature_extraction import detect_dataset_format, adapt_dataframe, adapt_cicflowmeter

    if df_raw.empty:
        st.error("Le fichier ne contient aucune donnee.")
        return None

    fmt, matched, missing, fmt_type = detect_dataset_format(df_raw, session_features)

    if fmt == "CIC-Darknet2020":
        return adapt_dataframe(df_raw, session_features) if missing else df_raw

    elif fmt == "CICFlowMeter":
        return adapt_cicflowmeter(df_raw, session_features)

    elif fmt_type == "packet_level":
        from src.csv_aggregation import detect_packet_csv, aggregate_packets_to_sessions
        is_packet, col_map, _ = detect_packet_csv(df_raw)
        if is_packet:
            df, _ = aggregate_packets_to_sessions(df_raw, col_map, session_features)
            if not df.empty:
                return df
        st.error("Impossible d'agreger les paquets en sessions.")
        return None

    elif fmt in ("Compatible (partiel)", "Partiellement compatible"):
        st.warning(
            f"**{len(matched)}/{len(session_features)} features** reconnues. "
            "Les features manquantes sont mises a zero — les predictions seront moins fiables."
        )
        return adapt_dataframe(df_raw, session_features)

    else:
        st.error(
            "**Format non reconnu.** Les formats supportes sont : "
            "CSV CIC-Darknet2020, CICFlowMeter, Wireshark CSV, PCAP/PCAPNG, Excel, Zeek conn.log."
        )
        return None


def _run_ext_predictions(df, models, session_features, config):
    """Lance les predictions du modele RF sur les donnees externes."""
    if "rf_session" not in models:
        st.error("Modele Random Forest non disponible.")
        return

    # Verifier que les features sont presentes
    missing_feats = [f for f in session_features if f not in df.columns]
    if missing_feats:
        st.warning(
            f"{len(missing_feats)} feature(s) manquante(s) dans les donnees. "
            "Elles sont remplacees par des zeros."
        )
        for f in missing_feats:
            df[f] = 0.0

    X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)
    probas = models["rf_session"].predict_proba(X)[:, 1]
    preds = (probas >= config["threshold"]).astype(int)

    st.session_state["ext_X"] = X
    st.session_state["ext_probas"] = probas
    st.session_state["ext_preds"] = preds

    if "label" in df.columns:
        y_true = df["label"].values.astype(int)
        st.session_state["ext_y_true"] = y_true
    else:
        # Supprimer si existait d'une session precedente
        st.session_state.pop("ext_y_true", None)


# =========================================================================
# PHASE 3 — AFFICHAGE DES RESULTATS
# =========================================================================

def _render_results(models, session_features, config):
    """Affiche les resultats de l'analyse externe."""
    df = st.session_state["ext_data"]
    source = st.session_state.get("ext_source", "")
    probas = st.session_state["ext_probas"]
    preds = st.session_state["ext_preds"]

    # --- En-tete ---
    col_title, col_source, col_reset = st.columns([3, 5, 2])
    with col_title:
        st.markdown("## Test externe")
    with col_source:
        st.caption(f"Source : {source}")
    with col_reset:
        if st.button("Nouvelle analyse", use_container_width=True, key="ext_reset"):
            _reset_ext_state()
            st.rerun()

    st.markdown("---")

    # --- A. Cartes metriques ---
    _render_metric_cards(df, probas, preds, session_features)

    st.markdown("---")

    # --- B / C : avec ou sans labels ---
    has_labels = "ext_y_true" in st.session_state

    if has_labels:
        _render_with_labels(probas, preds, config)
    else:
        _render_without_labels(df, probas, preds, config, session_features)

    st.markdown("---")

    # --- D. Export ---
    _render_export(df, probas, preds)


def _render_metric_cards(df, probas, preds, session_features):
    """Affiche les 4 cartes metriques de synthese."""
    n_total = len(df)
    n_alerts = int(preds.sum())
    alert_rate = n_alerts / max(n_total, 1)

    # Qualite des features
    n_features_available = sum(
        1 for f in session_features
        if f in df.columns and (df[f] != 0).any()
    )
    n_features_total = len(session_features)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Sessions analysees", f"{n_total:,}", "blue")
    with col2:
        render_metric_card("Alertes", f"{n_alerts:,}", "red" if n_alerts > 0 else "green")
    with col3:
        render_metric_card("Taux d'alerte", f"{alert_rate:.2%}", "yellow")
    with col4:
        fq_color = "green" if n_features_available >= 24 else ("yellow" if n_features_available >= 18 else "red")
        render_metric_card("Qualite features", f"{n_features_available}/{n_features_total}", fq_color)


# =========================================================================
# B — AVEC LABELS
# =========================================================================

def _render_with_labels(probas, preds, config):
    """Affiche les resultats quand les labels sont disponibles."""
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score,
        f1_score, roc_curve, auc,
    )

    y_true = st.session_state["ext_y_true"]

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    # --- Matrice de confusion ---
    col_cm, col_metrics = st.columns(2)

    with col_cm:
        st.subheader("Matrice de confusion")
        explain(
            "La matrice montre les 4 cas possibles : predictions correctes (diagonale) "
            "et erreurs (hors diagonale). <strong>TN</strong> et <strong>TP</strong> = le modele a raison. "
            "<strong>FP</strong> = fausse alerte. <strong>FN</strong> = menace ratee."
        )

        z = [[tn, fp], [fn, tp]]
        text = [
            [f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
            [f"FN<br>{fn:,}", f"TP<br>{tp:,}"],
        ]

        fig_cm = go.Figure(data=go.Heatmap(
            z=z,
            x=["Predit Benin", "Predit Malveillant"],
            y=["Reel Benin", "Reel Malveillant"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale=[[0, "#1e293b"], [1, "#3b82f6"]],
            showscale=False,
        ))
        fig_cm.update_layout(template="plotly_dark", height=350, margin=dict(t=30))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_metrics:
        st.subheader("Metriques de performance")
        explain(
            "<strong>Accuracy</strong> = % global de bonnes reponses. "
            "<strong>Precision</strong> = parmi les alertes, combien sont de vrais malwares. "
            "<strong>Recall</strong> = parmi les vrais malwares, combien sont detectes. "
            "<strong>F1</strong> = equilibre entre precision et recall."
        )
        render_metric_card("Accuracy", f"{100 * acc:.2f}%", "blue")
        render_metric_card("Precision", f"{100 * prec:.2f}%", "green")
        render_metric_card("Recall", f"{100 * rec:.2f}%", "yellow")
        render_metric_card("F1-score", f"{f1:.4f}", "blue")

    st.markdown("---")

    # --- Tableau de comparaison (la fonctionnalite cle) ---
    st.subheader("Comparaison avec le dataset d'entrainement")
    explain(
        "Ce tableau compare les performances du modele sur le <strong>dataset d'entrainement original</strong> "
        "(122 132 sessions) avec celles obtenues sur le <strong>dataset externe</strong> que vous venez de tester. "
        "Un ecart faible indique que le modele generalise bien ; un ecart superieur a 2% "
        "signale un possible surapprentissage."
    )

    ext_acc = 100 * acc
    ext_prec = 100 * prec
    ext_rec = 100 * rec
    ext_f1 = f1
    ext_fn = int(fn)
    ext_fp = int(fp)

    comparison_data = {
        "Metrique": ["Accuracy", "Precision", "Recall", "F1-score", "Faux negatifs (FN)", "Faux positifs (FP)"],
        "Dataset original": [
            f"{TRAINING_PERF['accuracy']:.2f}%",
            f"{TRAINING_PERF['precision']:.2f}%",
            f"{TRAINING_PERF['recall']:.2f}%",
            f"{TRAINING_PERF['f1']:.4f}",
            f"{TRAINING_PERF['fn']:,}",
            f"{TRAINING_PERF['fp']:,}",
        ],
        "Dataset externe": [
            f"{ext_acc:.2f}%",
            f"{ext_prec:.2f}%",
            f"{ext_rec:.2f}%",
            f"{ext_f1:.4f}",
            f"{ext_fn:,}",
            f"{ext_fp:,}",
        ],
        "Difference": [
            _format_diff(ext_acc - TRAINING_PERF["accuracy"]),
            _format_diff(ext_prec - TRAINING_PERF["precision"]),
            _format_diff(ext_rec - TRAINING_PERF["recall"]),
            _format_diff_f1(ext_f1 - TRAINING_PERF["f1"]),
            _format_diff_int(ext_fn, TRAINING_PERF["fn"]),
            _format_diff_int(ext_fp, TRAINING_PERF["fp"]),
        ],
    }

    df_comp = pd.DataFrame(comparison_data)

    # Styler pour mettre en evidence les differences
    def _highlight_diff(val):
        """Colore les differences : vert si favorable, rouge si defavorable."""
        if not isinstance(val, str) or val == "-":
            return ""
        try:
            # Extraire la valeur numerique
            num_str = val.replace("%", "").replace("+", "").replace(" ", "").replace("\u00a0", "")
            num = float(num_str)
        except (ValueError, TypeError):
            return ""
        # Pour les metriques principales (accuracy, precision, recall, f1) :
        # negatif = degradation = rouge. Pour FN/FP, positif = plus d'erreurs = rouge.
        return ""

    st.dataframe(
        df_comp.style.set_properties(**{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # --- Verdict de generalisation ---
    st.subheader("Verdict de generalisation")

    acc_drop = TRAINING_PERF["accuracy"] - ext_acc
    f1_drop = TRAINING_PERF["f1"] - ext_f1

    if acc_drop <= 1.0 and f1_drop <= 0.01:
        st.success(
            f"**Excellente generalisation.** "
            f"La chute d'accuracy est de seulement **{acc_drop:+.2f}%** et celle du F1 de **{f1_drop:+.4f}**. "
            f"Le modele conserve des performances quasi identiques sur ces nouvelles donnees."
        )
    elif acc_drop <= 2.0 and f1_drop <= 0.02:
        st.info(
            f"**Bonne generalisation.** "
            f"La chute d'accuracy est de **{acc_drop:+.2f}%** (F1 : **{f1_drop:+.4f}**). "
            f"Les performances restent proches du dataset original. "
            f"Ecart acceptable pour un deploiement en production."
        )
    elif acc_drop <= 5.0:
        st.warning(
            f"**Generalisation moderee.** "
            f"La chute d'accuracy est de **{acc_drop:+.2f}%** (F1 : **{f1_drop:+.4f}**). "
            f"Le modele montre des signes de surapprentissage. "
            f"Recommandation : reentrainer avec un dataset plus diversifie."
        )
    else:
        st.error(
            f"**Generalisation insuffisante.** "
            f"La chute d'accuracy est de **{acc_drop:+.2f}%** (F1 : **{f1_drop:+.4f}**). "
            f"Le modele ne generalise pas correctement sur ces donnees. "
            f"Causes possibles : distribution tres differente, features manquantes, ou surapprentissage. "
            f"Recommandation : verifier la compatibilite du dataset et reentrainer le modele."
        )

    st.markdown("---")

    # --- Courbe ROC ---
    st.subheader("Courbe ROC — Dataset externe")
    explain(
        "La courbe ROC montre la capacite du modele a distinguer benin vs malveillant "
        "a differents seuils sur le dataset externe. L'<strong>AUC</strong> (aire sous la courbe) "
        "vaut 1.0 pour un modele parfait et 0.5 pour un modele aleatoire."
    )

    fpr, tpr, _ = roc_curve(y_true, probas)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"Dataset externe (AUC = {roc_auc:.4f})",
        line=dict(color="#3b82f6", width=2.5),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Modele aleatoire (AUC = 0.5)",
        line=dict(color="gray", dash="dash"),
    ))
    fig_roc.update_layout(
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (TPR / Recall)",
        template="plotly_dark",
        height=400,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_roc, use_container_width=True)


# =========================================================================
# C — SANS LABELS
# =========================================================================

def _render_without_labels(df, probas, preds, config, session_features):
    """Affiche les resultats quand aucun label n'est disponible."""

    st.info(
        "Les donnees importees ne contiennent pas de labels (verite terrain). "
        "La comparaison avec le dataset d'entrainement n'est pas possible. "
        "Voici une analyse basee sur les predictions du modele uniquement."
    )

    # --- Distribution des probabilites ---
    st.subheader("Distribution des probabilites de malveillance")
    explain(
        "Distribution des scores P(malveillant) attribues par le modele. "
        "Un modele confiant produit une distribution bimodale (pics pres de 0 et 1). "
        "Les sessions dans la zone grise (0.3-0.7) meritent une attention particuliere."
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=probas, nbinsx=50,
        name="Toutes les sessions",
        marker_color="#3b82f6", opacity=0.7,
    ))
    fig_hist.add_vline(
        x=config["threshold"], line_dash="dash", line_color="red",
        annotation_text=f"Seuil = {config['threshold']}",
    )
    fig_hist.update_layout(
        xaxis_title="Probabilite de malveillance",
        yaxis_title="Nombre de sessions",
        template="plotly_dark",
        height=350,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # --- Alertes a differents seuils ---
    st.subheader("Nombre d'alertes selon le seuil")
    explain(
        "Ce graphique montre comment le nombre d'alertes varie en fonction du seuil de decision. "
        "Un seuil bas genere plus d'alertes (plus sensible), un seuil haut en genere moins (plus specifique)."
    )

    thresholds = np.arange(0.1, 1.0, 0.05)
    n_alerts_per_thresh = [int((probas >= t).sum()) for t in thresholds]

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Bar(
        x=[f"{t:.2f}" for t in thresholds],
        y=n_alerts_per_thresh,
        marker_color=["#ef4444" if t == config["threshold"] else "#3b82f6" for t in thresholds],
        name="Alertes",
    ))
    fig_thresh.update_layout(
        xaxis_title="Seuil de detection",
        yaxis_title="Nombre d'alertes",
        template="plotly_dark",
        height=350,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

    st.markdown("---")

    # --- Top 20 sessions suspectes ---
    st.subheader("Top 20 sessions les plus suspectes")
    explain(
        "Les 20 sessions avec la probabilite de malveillance la plus elevee. "
        "Examinez ces sessions en priorite pour determiner s'il s'agit de vrais incidents."
    )

    df_display = df.copy()
    df_display["probabilite"] = probas
    df_display["verdict"] = np.where(preds == 1, "SUSPECT", "Benin")

    # Selectionner les colonnes a afficher
    display_cols = ["probabilite", "verdict"]
    for col in ["unique_link_mark", "Src IP", "Dst IP", "src_ip", "dst_ip",
                 "id.orig_h", "id.resp_h"]:
        if col in df_display.columns:
            display_cols.append(col)

    # Ajouter les top 5 features presentes
    top_feats = [f for f in session_features[:5] if f in df_display.columns]
    display_cols.extend(top_feats)

    # Supprimer les doublons tout en preservant l'ordre
    seen = set()
    unique_cols = []
    for c in display_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    display_cols = unique_cols

    available_cols = [c for c in display_cols if c in df_display.columns]
    df_top = df_display[available_cols].sort_values("probabilite", ascending=False).head(20)

    styled = df_top.style.background_gradient(
        subset=["probabilite"], cmap="RdYlGn_r", vmin=0, vmax=1
    )
    st.dataframe(styled, use_container_width=True, height=500)


# =========================================================================
# D — EXPORT
# =========================================================================

def _render_export(df, probas, preds):
    """Affiche le bouton d'export CSV des resultats."""
    st.subheader("Exporter les resultats")

    df_export = df.copy()
    df_export["probabilite_malveillance"] = probas
    df_export["prediction"] = np.where(preds == 1, "Malveillant", "Benin")

    if "ext_y_true" in st.session_state:
        y_true = st.session_state["ext_y_true"]
        df_export["label_reel"] = y_true
        df_export["correct"] = np.where(preds == y_true, "Oui", "Non")

    csv_data = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Telecharger les resultats (CSV)",
        data=csv_data,
        file_name="micro_siem_test_externe_resultats.csv",
        mime="text/csv",
        use_container_width=True,
        key="ext_download_csv",
    )


# =========================================================================
# E — RESET
# =========================================================================

def _reset_ext_state():
    """Supprime toutes les variables ext_ de session_state."""
    keys_to_remove = [k for k in st.session_state if k.startswith("ext_")]
    for k in keys_to_remove:
        del st.session_state[k]


# =========================================================================
# UTILITAIRES
# =========================================================================

def _format_diff(diff_pct):
    """Formate une difference en pourcentage avec signe."""
    if abs(diff_pct) < 0.005:
        return "0.00%"
    sign = "+" if diff_pct > 0 else ""
    return f"{sign}{diff_pct:.2f}%"


def _format_diff_f1(diff_f1):
    """Formate une difference de F1-score avec signe."""
    if abs(diff_f1) < 0.00005:
        return "0.0000"
    sign = "+" if diff_f1 > 0 else ""
    return f"{sign}{diff_f1:.4f}"


def _format_diff_int(ext_val, train_val):
    """Formate la difference pour les entiers (FN, FP) — non comparable directement
    car les tailles de dataset different, donc on affiche un ratio normalise."""
    ext_total = len(st.session_state.get("ext_data", []))
    train_total = TRAINING_PERF["total_test"]
    if ext_total == 0 or train_total == 0:
        return "-"
    # Taux d'erreur normalise pour comparaison
    ext_rate = 100 * ext_val / ext_total
    train_rate = 100 * train_val / train_total
    diff = ext_rate - train_rate
    if abs(diff) < 0.005:
        return "~0.00%"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.2f}% (taux)"
