"""
Page 5 : Configuration — Visualisation de l'impact du seuil et de l'IF.
Les parametres sont ajustables via la sidebar (panneau de gauche).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card, has_labels


def render(config):
    st.header("Configuration de la detection")
    st.markdown("Ajustez le **seuil** et l'**Isolation Forest** via la sidebar (panneau de gauche).")

    from src.ui_components import require_data
    if not require_data("Chargez des donnees pour voir l'impact des parametres de detection."):
        return config

    threshold = config["threshold"]
    probas = st.session_state["probas"]
    n_alerts = int((probas >= threshold).sum())
    n_total = len(probas)

    st.markdown("---")
    st.subheader("Seuil de decision")

    explain(
        "Le seuil determine a partir de quelle probabilite une session est consideree suspecte. "
        "Modifiez-le dans la <strong>sidebar</strong> (panneau de gauche). "
        "<strong>Seuil bas</strong> (ex: 0.3) = plus d'alertes, moins de malwares rates, "
        "mais plus de fausses alertes. "
        "<strong>Seuil haut</strong> (ex: 0.7) = moins d'alertes, mais risque de rater des menaces."
    )

    rec_threshold = st.session_state.get("recommended_threshold", 0.5)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        render_metric_card("Seuil actuel", f"{threshold}", "blue")
    with col_t2:
        render_metric_card("Seuil recommande", f"{rec_threshold}", "green" if threshold == rec_threshold else "yellow")
    with col_t3:
        render_metric_card("Taux d'alerte", f"{100*n_alerts/n_total:.1f}%", "yellow")

    if threshold != rec_threshold:
        fq = st.session_state.get("feature_quality")
        fq_avail = fq.get("available", 27) if fq else 27
        if fq_avail < 24:
            explain(
                f"Avec <strong>{fq_avail}/27 features</strong>, le seuil recommande est "
                f"<strong>{rec_threshold}</strong> (plus bas pour compenser les features manquantes)."
            )

    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Alertes avec ce seuil", f"{n_alerts:,} / {n_total:,}", "red")
    with col2:
        mean_conf = float(st.session_state.get("confidence", np.array([0.5])).mean())
        conf_color = "green" if mean_conf >= 0.7 else ("yellow" if mean_conf >= 0.5 else "red")
        render_metric_card("Confiance moyenne", f"{mean_conf:.0%}", conf_color)

    # Ground truth si disponible
    if has_labels():
        y_true = st.session_state["y_true"]
        preds_t = (probas >= threshold).astype(int)
        from sklearn.metrics import confusion_matrix as cm
        tn, fp, fn, tp = cm(y_true, preds_t).ravel()

        st.markdown("---")
        st.subheader("Impact sur les erreurs")

        explain(
            "<strong>Vrais positifs (TP)</strong> = malwares correctement detectes. "
            "<strong>Faux positifs (FP)</strong> = sessions benignes alertees par erreur. "
            "<strong>Faux negatifs (FN)</strong> = malwares rates (le plus dangereux). "
            "<strong>Vrais negatifs (TN)</strong> = sessions benignes correctement ignorees."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Vrais positifs (TP)", f"{tp:,}", "green")
        with col2:
            render_metric_card("Faux positifs (FP)", f"{fp:,}", "yellow")
        with col3:
            render_metric_card("Faux negatifs (FN)", f"{fn:,}", "red")
        with col4:
            render_metric_card("Vrais negatifs (TN)", f"{tn:,}", "green")

        # Courbe FN/FP + F1 par seuil
        st.markdown("---")
        st.subheader("Optimisation du seuil")

        explain(
            "Ces graphiques montrent comment le seuil impacte les erreurs et le F1-score. "
            "Le <strong>F1-score</strong> (equilibre precision/recall) est maximal au seuil optimal. "
            "Les faux negatifs (menaces ratees, en rouge) sont les plus dangereux."
        )

        from sklearn.metrics import f1_score as f1_metric, precision_score as prec_metric, recall_score as rec_metric
        thresholds_range = np.arange(0.05, 0.95, 0.05)
        fns, fps, f1s, precs, recs = [], [], [], [], []
        for t in thresholds_range:
            p = (probas >= t).astype(int)
            tn_t, fp_t, fn_t, tp_t = cm(y_true, p).ravel()
            fns.append(fn_t)
            fps.append(fp_t)
            f1s.append(f1_metric(y_true, p, zero_division=0))
            precs.append(prec_metric(y_true, p, zero_division=0))
            recs.append(rec_metric(y_true, p, zero_division=0))

        # Seuil optimal (max F1)
        best_idx = np.argmax(f1s)
        best_threshold = thresholds_range[best_idx]
        best_f1 = f1s[best_idx]

        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresholds_range, y=fns,
                                     name="Faux negatifs (menaces ratees)",
                                     line=dict(color="#ef4444", width=2.5)))
            fig.add_trace(go.Scatter(x=thresholds_range, y=fps,
                                     name="Faux positifs (fausses alertes)",
                                     line=dict(color="#f59e0b", width=2.5)))
            fig.add_vline(x=threshold, line_dash="dash", line_color="white",
                          annotation_text=f"Actuel = {threshold}")
            fig.add_vline(x=best_threshold, line_dash="dot", line_color="#10b981",
                          annotation_text=f"Optimal = {best_threshold:.2f}")
            fig.update_layout(
                xaxis_title="Seuil de decision",
                yaxis_title="Nombre d'erreurs",
                template="plotly_dark", height=350,
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_opt2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=thresholds_range, y=f1s,
                                      name="F1-score",
                                      line=dict(color="#3b82f6", width=2.5)))
            fig2.add_trace(go.Scatter(x=thresholds_range, y=precs,
                                      name="Precision",
                                      line=dict(color="#10b981", width=1.5, dash="dot")))
            fig2.add_trace(go.Scatter(x=thresholds_range, y=recs,
                                      name="Recall",
                                      line=dict(color="#f59e0b", width=1.5, dash="dot")))
            fig2.add_vline(x=threshold, line_dash="dash", line_color="white",
                          annotation_text=f"Actuel = {threshold}")
            fig2.add_vline(x=best_threshold, line_dash="dot", line_color="#10b981",
                          annotation_text=f"Optimal = {best_threshold:.2f}")
            fig2.update_layout(
                xaxis_title="Seuil de decision",
                yaxis_title="Score (0-1)",
                template="plotly_dark", height=350,
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            f"Seuil optimal (max F1) : **{best_threshold:.2f}** (F1 = {best_f1:.4f}). "
            f"Seuil actuel : **{threshold}**."
        )

    else:
        st.markdown("---")
        st.subheader("Metriques d'erreur")
        st.info(
            "Les metriques d'erreur (faux positifs, faux negatifs) necessitent des **labels** "
            "(verite terrain), c'est-a-dire savoir a l'avance quelles sessions sont reellement malveillantes. "
            "Le trafic reel importe via PCAP n'a pas de labels — c'est normal.\n\n"
            "**Ce qui reste disponible sans labels :**\n"
            "- Le nombre d'alertes ci-dessus evolue en temps reel quand vous deplacez le seuil dans la sidebar\n"
            "- L'Isolation Forest (ci-dessous) detecte les sessions anormales independamment du seuil\n"
            "- La page **Vue d'ensemble** applique les parametres choisis ici"
        )

    st.markdown("---")

    st.subheader("Isolation Forest (detection d'anomalies)")

    explain(
        "L'Isolation Forest est un modele <strong>non supervise</strong> : il a appris ce qu'est "
        "du trafic normal sans jamais voir de malware. Il detecte les sessions qui s'ecartent de la norme. "
        "Si active, une session est alertee si le Random Forest <strong>ou</strong> l'IF la signale. "
        "Activez-le dans la <strong>sidebar</strong> (panneau de gauche)."
    )

    if_status = "Active" if config["use_if"] else "Desactive"
    if_color = "green" if config["use_if"] else "blue"
    render_metric_card("Isolation Forest", if_status, if_color)

    # Metriques IF si actif
    if config["use_if"] and st.session_state.get("if_preds") is not None:
        if_preds = st.session_state["if_preds"]
        n_if = int(if_preds.sum())
        n_if_only = int(((if_preds == 1) & ((probas >= threshold) == False)).sum())

        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("Anomalies IF detectees", f"{n_if:,}", "yellow")
        with col2:
            render_metric_card("Ajoutees par IF (hors RF)", f"{n_if_only:,}", "yellow")

        explain(
            f"L'IF signale <strong>{n_if:,}</strong> sessions comme anomalies. "
            f"Parmi elles, <strong>{n_if_only:,}</strong> ne sont pas deja en alerte par le RF. "
            f"Ces sessions sont potentiellement des menaces non detectees par le classificateur supervise."
        )

    return config
