"""
Page 5 : Configuration — Seuil de detection, toggle IF, courbe FN/FP.
"""

import streamlit as st
import numpy as np

from src.ui_components import explain, render_metric_card


def render(config):
    st.header("Configuration de la detection")
    st.markdown("Ajustez les parametres de detection et observez l'impact en temps reel.")

    st.markdown("---")
    st.subheader("Seuil de decision")

    explain(
        "Le seuil determine a partir de quelle probabilite une session est consideree suspecte. "
        "<strong>Seuil bas</strong> (ex: 0.3) = plus d'alertes, moins de malwares rates, "
        "mais plus de fausses alertes. "
        "<strong>Seuil haut</strong> (ex: 0.7) = moins d'alertes, mais risque de rater des menaces. "
        "Le seuil par defaut (0.5) est un compromis raisonnable."
    )

    threshold = st.slider(
        "Seuil de detection",
        min_value=0.0, max_value=1.0, value=config["threshold"], step=0.01,
    )

    # Afficher l'impact en temps reel
    if "probas" in st.session_state:
        probas = st.session_state["probas"]
        n_alerts = int((probas >= threshold).sum())
        n_total = len(probas)

        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("Alertes avec ce seuil", f"{n_alerts:,} / {n_total:,}", "red")
        with col2:
            render_metric_card("Taux d'alerte", f"{100*n_alerts/n_total:.1f}%", "yellow")

        # Ground truth si disponible
        if "y_true" in st.session_state:
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

            # Courbe FN/FP par seuil
            st.markdown("---")
            st.subheader("Compromis FN / FP en fonction du seuil")

            explain(
                "Ce graphique montre comment le nombre de faux negatifs (menaces ratees, en rouge) "
                "et de faux positifs (fausses alertes, en orange) evolue quand vous deplacez le seuil. "
                "L'objectif est de trouver le point ou les deux courbes sont au plus bas."
            )

            import plotly.graph_objects as go
            thresholds_range = np.arange(0.05, 0.95, 0.05)
            fns, fps = [], []
            for t in thresholds_range:
                p = (probas >= t).astype(int)
                tn_t, fp_t, fn_t, tp_t = cm(y_true, p).ravel()
                fns.append(fn_t)
                fps.append(fp_t)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresholds_range, y=fns,
                                     name="Faux negatifs (menaces ratees)",
                                     line=dict(color="#ef4444", width=2.5)))
            fig.add_trace(go.Scatter(x=thresholds_range, y=fps,
                                     name="Faux positifs (fausses alertes)",
                                     line=dict(color="#f59e0b", width=2.5)))
            fig.add_vline(x=threshold, line_dash="dash", line_color="white",
                          annotation_text=f"Seuil actuel = {threshold}")
            fig.update_layout(
                xaxis_title="Seuil de decision",
                yaxis_title="Nombre d'erreurs",
                template="plotly_dark", height=400,
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Isolation Forest (detection d'anomalies)")

    explain(
        "L'Isolation Forest est un modele <strong>non supervise</strong> : il a appris ce qu'est "
        "du trafic normal sans jamais voir de malware. Il detecte les sessions qui s'ecartent de la norme. "
        "Si active, une session est alertee si le Random Forest <strong>ou</strong> l'IF la signale."
    )

    use_if = st.toggle(
        "Activer l'Isolation Forest en complement du Random Forest",
        value=config["use_if"],
    )

    return {"threshold": threshold, "use_if": use_if}
