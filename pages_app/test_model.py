"""
Page Tester le modele — Selection de dataset, predictions, performance, limites.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Datasets disponibles (trafic chiffre uniquement) ===
DATASETS = {
    "CIC-Darknet2020 — 1 000 sessions (test set, mix 50/50)": {
        "file": "external_test_sample.csv",
        "label": "CIC-Darknet2020 — 1 000 sessions (500 benignes + 500 malveillantes)",
        "features": 27,
        "source": "CIC-Darknet2020 (Univ. New Brunswick, 2020)",
        "attacks": "25 familles de malwares (trojans, ransomware, botnets)",
        "encrypted": True,
    },
    "CIC-Darknet2020 — 5 000 sessions (distribution reelle)": {
        "file": "demo_sample.csv",
        "label": "CIC-Darknet2020 — 5 000 sessions (distribution reelle du dataset)",
        "features": 27,
        "source": "CIC-Darknet2020 (Univ. New Brunswick, 2020)",
        "attacks": "25 familles de malwares (trojans, ransomware, botnets)",
        "encrypted": True,
    },
    "HIKARI-2021 — 1 000 sessions (trafic HTTPS : Probing, Bruteforce, CryptoMiner)": {
        "file": "sample_hikari2021.csv",
        "label": "HIKARI-2021 — 1 000 sessions chiffrees (500 benignes + 500 attaques)",
        "features": 19,
        "source": "HIKARI-2021 (Zenodo/MDPI, 2021)",
        "attacks": "Probing, Bruteforce HTTPS/XML, CryptoMiner XMRIGCC",
        "encrypted": True,
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
        "Selectionnez un dataset de trafic chiffre pour evaluer les performances du modele "
        "Random Forest. Les resultats sont compares aux performances obtenues a l'entrainement "
        "(F1 = 0.995 sur CIC-Darknet2020)."
    )

    # === Auto-load ===
    if "data" not in st.session_state or "probas" not in st.session_state:
        default = DATASETS[list(DATASETS.keys())[0]]
        _load_csv_dataset(default["file"], default["label"], default["features"], session_features)
        st.session_state["_loaded_dataset"] = default["file"]

    if "data" not in st.session_state or "probas" not in st.session_state:
        st.error("Impossible de charger le dataset par defaut.")
        return

    # === Selecteur ===
    selected = st.selectbox(
        "Choisir un dataset de test",
        list(DATASETS.keys()),
        key="dataset_selector",
    )
    ds = DATASETS[selected]

    if st.session_state.get("_loaded_dataset") != ds["file"]:
        _load_csv_dataset(ds["file"], ds["label"], ds["features"], session_features)
        st.session_state["_loaded_dataset"] = ds["file"]
        st.rerun()

    st.caption(
        f"Source : {ds['source']} | Attaques : {ds['attacks']} | "
        f"Features : {ds['features']}/27"
    )

    # === Resultats ===
    _render_results(models, session_features, config, ds)

    # === Import custom ===
    st.markdown("---")
    with st.expander("Importer vos propres donnees", expanded=False):
        st.caption("PCAP/PCAPNG, CSV (Wireshark, CICFlowMeter, tshark), Excel, Zeek conn.log")
        uploaded = st.file_uploader(
            "Importer un fichier",
            type=["pcap", "pcapng", "csv", "xlsx", "xls", "log", "tsv"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            if _process_upload(uploaded, session_features):
                st.session_state["_loaded_dataset"] = "__custom__"
                st.rerun()


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


# =========================================================================
# AFFICHAGE DES RESULTATS
# =========================================================================

def _render_results(models, session_features, config, ds_info):
    """Affiche resultats complets."""
    df = st.session_state["data"]
    probas = st.session_state["probas"]
    threshold = config["threshold"]
    preds = (probas >= threshold).astype(int)
    has_labels = "label" in df.columns

    if has_labels:
        y_true = df["label"].values.astype(int)

    n_total = len(df)
    n_alerts = int(preds.sum())
    n_high = int((probas >= 0.8).sum())

    # --- Metriques rapides ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Sessions", f"{n_total:,}", "blue")
    with col2:
        render_metric_card("Benignes", f"{n_total - n_alerts:,}", "green")
    with col3:
        render_metric_card("Alertes", f"{n_alerts:,}", "red" if n_alerts > 0 else "green")
    with col4:
        render_metric_card("Critiques (P>0.8)", f"{n_high:,}", "yellow")

    # --- Performance (si labels) ---
    if has_labels:
        _render_performance(preds, probas, y_true, ds_info)

    # --- Distribution des probabilites ---
    st.markdown("---")
    st.subheader("Distribution des probabilites")

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
    fig.update_layout(
        xaxis_title="Probabilite (0 = benin, 1 = malveillant)",
        yaxis_title="Nombre de sessions",
        template="plotly_dark", height=350, margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Top sessions suspectes ---
    st.subheader("Sessions les plus suspectes")
    df_display = df.copy()
    df_display["probabilite"] = probas
    df_display["verdict"] = np.where(preds == 1, "SUSPECT", "Benin")
    if has_labels:
        df_display["verite_terrain"] = np.where(y_true == 1, "Malveillant", "Benin")

    display_cols = ["probabilite", "verdict"]
    if has_labels:
        display_cols.append("verite_terrain")

    df_sorted = df_display[display_cols].sort_values("probabilite", ascending=False)
    styled = df_sorted.head(100).style.background_gradient(
        subset=["probabilite"], cmap="RdYlGn_r", vmin=0, vmax=1
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # --- Export ---
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


def _render_performance(preds, probas, y_true, ds_info):
    """Performance test vs entrainement + matrice de confusion + encart limites."""
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

    # Note si features incompletes
    n_feat = ds_info.get("features", 27)
    if n_feat < 27:
        explain(
            f"<strong>Attention</strong> : ce dataset dispose de <strong>{n_feat}/27 features</strong>. "
            f"Les features manquantes (TTL, TCP headers) sont mises a zero, "
            f"ce qui degrade les performances."
        )

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
        explain(
            f"<strong>TP = {tp:,}</strong> malwares detectes.<br>"
            f"<strong>TN = {tn:,}</strong> sessions benignes correctes.<br>"
            f"<strong>FP = {fp:,}</strong> fausses alertes.<br>"
            f"<strong>FN = {fn:,}</strong> malwares rates."
        )

    # === Encart limites (si performance faible) ===
    if f1 < 0.5:
        st.markdown("---")
        st.subheader("Limites de generalisation")
        st.error(
            f"**F1 = {f1:.4f}** — Le modele ne detecte pas les attaques de ce dataset."
        )

        reasons = []
        if n_feat < 27:
            reasons.append(
                f"**Features incompletes** : seules {n_feat}/27 features sont disponibles. "
                f"Les 8 features manquantes (TTL, TCP headers, etc.) ne sont pas extractibles "
                f"depuis le format CICFlowMeter et sont mises a zero."
            )
        reasons.append(
            f"**Types d'attaques differents** : le modele a ete entraine sur du trafic darknet "
            f"(trojans, ransomware, botnets). Ce dataset contient des attaques de type "
            f"*{ds_info.get('attacks', 'inconnu')}*, qui ont des patterns reseau differents."
        )
        reasons.append(
            "**Specialisation du modele** : un modele Random Forest entraine sur un type "
            "specifique de trafic malveillant ne generalise pas automatiquement a d'autres "
            "types d'attaques. C'est une limitation connue des approches supervisees."
        )

        for i, r in enumerate(reasons, 1):
            st.markdown(f"{i}. {r}")

        explain(
            "Ce resultat est attendu et illustre l'importance de l'adequation entre "
            "les donnees d'entrainement et les donnees de test. "
            "Pour detecter d'autres types d'attaques, il faudrait re-entrainer le modele "
            "sur des donnees incluant ces types d'attaques."
        )
