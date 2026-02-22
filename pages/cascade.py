"""
Page 3 : Mode cascade — Analyse session puis paquets pour les sessions incertaines.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card
from src.models import load_rf_packet, DEMO_PACKETS_PATH
from src.cascade import identify_uncertain_sessions, cascade_analysis


def render(models, session_features, packet_features, config):
    st.header("Mode cascade : Session puis Paquets")

    explain(
        "Le mode cascade est une approche <strong>multi-granularite</strong>. "
        "Le Random Forest session analyse d'abord toutes les sessions (27 features). "
        "Pour celles dont la probabilite est <strong>incertaine</strong> (zone grise), "
        "on descend au niveau paquet individuel avec un second modele RF (21 features, 99.98% accuracy). "
        "Les predictions par paquet sont ensuite <strong>agregees</strong> pour donner un verdict final plus precis."
    )

    # Verifier que des donnees session sont chargees
    if "probas" not in st.session_state or "data" not in st.session_state:
        st.warning("Chargez d'abord des donnees dans **Vue d'ensemble** (cliquez sur 'Charger la demo').")
        return

    probas = st.session_state["probas"]
    df_session = st.session_state["data"]

    if "unique_link_mark" not in df_session.columns:
        st.error("Les donnees session ne contiennent pas de colonne `unique_link_mark` pour lier les paquets.")
        return

    # --- Configuration cascade ---
    st.markdown("---")
    st.subheader("Parametres de la cascade")

    col1, col2, col3 = st.columns(3)
    with col1:
        low = st.slider("Borne basse (incertitude)", 0.0, 1.0, 0.3, 0.05,
                         help="Sessions avec P >= cette valeur entrent dans la zone incertaine")
    with col2:
        high = st.slider("Borne haute (incertitude)", 0.0, 1.0, 0.7, 0.05,
                          help="Sessions avec P <= cette valeur restent dans la zone incertaine")
    with col3:
        strategy = st.radio("Strategie d'agregation", ["conservative", "vote", "mean_proba"],
                            help="Comment combiner les predictions paquet en verdict session")

    # Compter sessions incertaines
    uncertain_idx = identify_uncertain_sessions(probas, low, high)
    n_uncertain = len(uncertain_idx)
    n_total = len(probas)

    explain(
        f"Avec les bornes [{low:.2f}, {high:.2f}], <strong>{n_uncertain}</strong> sessions "
        f"sur {n_total} ({100*n_uncertain/n_total:.1f}%) sont dans la zone d'incertitude. "
        f"Ces sessions seront re-analysees au niveau paquet."
    )

    strategy_desc = {
        "conservative": "Si <strong>un seul paquet</strong> est predit malveillant, la session entiere est malveillante. Maximise le recall (moins de menaces ratees).",
        "vote": "<strong>Vote majoritaire</strong> : la session est malveillante si plus de 50% des paquets le sont. Equilibre precision/recall.",
        "mean_proba": "<strong>Moyenne des probabilites</strong> : si la probabilite moyenne des paquets >= 0.5, la session est malveillante. Plus nuance."
    }
    explain(f"Strategie <strong>{strategy}</strong> : {strategy_desc[strategy]}")

    if n_uncertain == 0:
        st.success("Aucune session dans la zone d'incertitude. Le modele session est confiant pour toutes les sessions.")
        return

    # --- Charger donnees paquets ---
    st.markdown("---")
    st.subheader("Donnees paquets")

    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        pkt_uploaded = st.file_uploader(
            "Importer un CSV de paquets (avec unique_link_mark)",
            type=["csv"],
            key="cascade_pkt_upload",
            help="CSV avec les 21 features packet-based et une colonne unique_link_mark."
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        load_pkt_demo = st.button("Charger demo paquets",
                                   use_container_width=True, type="primary")

    if pkt_uploaded is not None:
        df_packets = pd.read_csv(pkt_uploaded, low_memory=False)
        st.session_state["packet_data"] = df_packets
        st.session_state["packet_source"] = pkt_uploaded.name
    elif load_pkt_demo:
        if os.path.exists(DEMO_PACKETS_PATH):
            df_packets = pd.read_csv(DEMO_PACKETS_PATH, low_memory=False)
            st.session_state["packet_data"] = df_packets
            st.session_state["packet_source"] = "Demo : paquets du dataset CIC-Darknet2020"
        else:
            st.error("Fichier de demonstration paquets introuvable.")
            return

    if "packet_data" not in st.session_state:
        st.info("Importez un CSV de paquets ou chargez la demo pour lancer la cascade.")
        return

    df_packets = st.session_state["packet_data"]
    st.caption(f"Source paquets : {st.session_state.get('packet_source', '')}")
    st.caption(f"{len(df_packets):,} paquets de {df_packets['unique_link_mark'].nunique()} sessions")

    # --- Charger modele RF paquets ---
    model_rf_pkt = load_rf_packet()
    if model_rf_pkt is None:
        st.error("Modele RF paquets non disponible (model_rf_paquets.joblib).")
        return

    # --- Lancer la cascade ---
    st.markdown("---")
    st.subheader("Resultats de la cascade")

    with st.spinner("Analyse cascade en cours..."):
        session_marks = df_session["unique_link_mark"].values
        results, uncertain_idx = cascade_analysis(
            probas, session_marks, df_packets,
            model_rf_pkt, packet_features,
            low=low, high=high, strategy=strategy
        )

    if not results:
        st.warning("Aucun paquet trouve pour les sessions incertaines. Verifiez que les donnees paquets correspondent aux sessions chargees.")
        return

    # --- Afficher les resultats ---
    n_resolved = len(results)
    n_changed_to_malicious = sum(1 for r in results.values() if r["verdict"] == 1)
    n_changed_to_benign = sum(1 for r in results.values() if r["verdict"] == 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Sessions incertaines", f"{n_uncertain}", "yellow")
    with col2:
        render_metric_card("Resolues par cascade", f"{n_resolved}", "blue")
    with col3:
        render_metric_card("Reclassees malveillantes", f"{n_changed_to_malicious}", "red")
    with col4:
        render_metric_card("Reclassees benignes", f"{n_changed_to_benign}", "green")

    explain(
        f"La cascade a analyse <strong>{n_resolved}</strong> sessions incertaines au niveau paquet. "
        f"Avec la strategie <strong>{strategy}</strong>, "
        f"<strong>{n_changed_to_malicious}</strong> ont ete reclassees comme malveillantes "
        f"et <strong>{n_changed_to_benign}</strong> comme benignes."
    )

    # --- Tableau detaille ---
    rows = []
    for sid, r in results.items():
        # Trouver l'index session correspondant
        session_idx_arr = np.where(session_marks == sid)[0]
        if len(session_idx_arr) == 0:
            continue
        session_idx = session_idx_arr[0]
        rows.append({
            "session_id": sid,
            "proba_session_RF": probas[session_idx],
            "verdict_cascade": "SUSPECT" if r["verdict"] == 1 else "Benin",
            "n_paquets": r["n_packets"],
            "paquets_malveillants": r["n_malicious_packets"],
            "proba_moyenne_paquets": r["mean_proba"],
            "proba_max_paquets": r["max_proba"],
            "confiance_cascade": r["confidence"],
        })

    df_results = pd.DataFrame(rows).sort_values("proba_session_RF", ascending=False)

    # Ajouter verite terrain si disponible
    if "y_true" in st.session_state:
        y_true = st.session_state["y_true"]
        mark_to_label = {session_marks[i]: y_true[i] for i in range(len(session_marks))}
        df_results["verite_terrain"] = df_results["session_id"].map(
            lambda s: "Malveillant" if mark_to_label.get(s, 0) == 1 else "Benin"
        )

    st.dataframe(
        df_results.style.background_gradient(
            subset=["proba_session_RF"], cmap="RdYlGn_r", vmin=0, vmax=1
        ).format({
            "proba_session_RF": "{:.4f}",
            "proba_moyenne_paquets": "{:.4f}",
            "proba_max_paquets": "{:.4f}",
            "confiance_cascade": "{:.4f}",
        }),
        use_container_width=True,
        height=400
    )

    explain(
        "Ce tableau montre chaque session incertaine et son nouveau verdict apres cascade. "
        "<strong>proba_session_RF</strong> = probabilite originale du RF session. "
        "<strong>paquets_malveillants</strong> = nombre de paquets predits malveillants dans la session. "
        "<strong>confiance_cascade</strong> = confiance du verdict final (depend de la strategie)."
    )

    # --- Graphique comparatif ---
    st.markdown("---")
    st.subheader("Comparaison avant / apres cascade")

    # Calcul des verdicts avant/apres
    threshold = config["threshold"]
    before_malicious = int((probas[uncertain_idx] >= threshold).sum())
    before_benign = n_uncertain - before_malicious

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avant cascade (RF session seul)",
        x=["Malveillant", "Benin"],
        y=[before_malicious, before_benign],
        marker_color=["#ef4444", "#10b981"],
        opacity=0.6
    ))
    fig.add_trace(go.Bar(
        name="Apres cascade (RF session + paquets)",
        x=["Malveillant", "Benin"],
        y=[n_changed_to_malicious, n_changed_to_benign],
        marker_color=["#ef4444", "#10b981"],
        opacity=1.0
    ))
    fig.update_layout(
        barmode="group",
        yaxis_title="Nombre de sessions",
        template="plotly_dark",
        height=350,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    explain(
        "Ce graphique compare les verdicts <strong>avant</strong> (RF session seul avec seuil) "
        "et <strong>apres</strong> la cascade (RF paquets). "
        "La cascade affine les predictions pour les sessions ou le modele session etait incertain."
    )

    # --- Performance si labels disponibles ---
    if "y_true" in st.session_state:
        st.markdown("---")
        st.subheader("Impact de la cascade sur les erreurs")

        y_true = st.session_state["y_true"]

        # Verdicts originaux
        original_preds = (probas >= threshold).astype(int)

        # Verdicts mis a jour
        updated_preds = original_preds.copy()
        for sid, r in results.items():
            idx_arr = np.where(session_marks == sid)[0]
            if len(idx_arr) > 0:
                updated_preds[idx_arr[0]] = r["verdict"]

        from sklearn.metrics import confusion_matrix
        tn1, fp1, fn1, tp1 = confusion_matrix(y_true, original_preds).ravel()
        tn2, fp2, fn2, tp2 = confusion_matrix(y_true, updated_preds).ravel()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Avant cascade (RF session seul)**")
            render_metric_card("Faux negatifs", f"{fn1}", "red")
            render_metric_card("Faux positifs", f"{fp1}", "yellow")
        with col2:
            st.markdown("**Apres cascade (RF session + paquets)**")
            color_fn = "green" if fn2 < fn1 else ("red" if fn2 > fn1 else "blue")
            color_fp = "green" if fp2 < fp1 else ("yellow" if fp2 > fp1 else "blue")
            render_metric_card("Faux negatifs", f"{fn2} ({fn2-fn1:+d})", color_fn)
            render_metric_card("Faux positifs", f"{fp2} ({fp2-fp1:+d})", color_fp)

        explain(
            f"La cascade a modifie <strong>{abs(fn2-fn1) + abs(fp2-fp1)}</strong> predictions. "
            f"Faux negatifs (menaces ratees) : {fn1} -> {fn2} ({fn2-fn1:+d}). "
            f"Faux positifs (fausses alertes) : {fp1} -> {fp2} ({fp2-fp1:+d})."
        )
