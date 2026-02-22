"""
Page 6 : Statistiques — Matrice de confusion, ROC, feature importance, scores IF.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card


def render(models, session_features, config):
    st.header("Statistiques du modele")
    st.markdown("Performance globale du modele sur les donnees chargees.")

    if "probas" not in st.session_state:
        st.warning("Chargez d'abord des donnees dans **Vue d'ensemble** (cliquez sur 'Charger la demo').")
        return

    probas = st.session_state["probas"]
    preds = st.session_state["preds"]

    # --- Matrice de confusion ---
    if "y_true" in st.session_state:
        y_true = st.session_state["y_true"]
        from sklearn.metrics import (confusion_matrix, accuracy_score,
                                     precision_score, recall_score, f1_score,
                                     roc_curve, auc)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Matrice de confusion")
            explain(
                "La matrice montre les 4 cas possibles : predictions correctes (diagonale) "
                "et erreurs (hors diagonale). <strong>TN</strong> et <strong>TP</strong> = le modele a raison. "
                "<strong>FP</strong> = fausse alerte. <strong>FN</strong> = menace ratee."
            )

            z = [[tn, fp], [fn, tp]]
            text = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
                    [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]

            fig = go.Figure(data=go.Heatmap(
                z=z, x=["Predit Benin", "Predit Malveillant"],
                y=["Reel Benin", "Reel Malveillant"],
                text=text, texttemplate="%{text}", textfont=dict(size=14),
                colorscale=[[0, "#1e293b"], [1, "#3b82f6"]], showscale=False
            ))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Metriques de performance")
            explain(
                "<strong>Accuracy</strong> = % global de bonnes reponses. "
                "<strong>Precision</strong> = parmi les alertes, combien sont de vrais malwares. "
                "<strong>Recall</strong> = parmi les vrais malwares, combien sont detectes. "
                "<strong>F1</strong> = equilibre entre precision et recall."
            )
            render_metric_card("Accuracy", f"{100*acc:.2f}%", "blue")
            render_metric_card("Precision", f"{100*prec:.2f}%", "green")
            render_metric_card("Recall", f"{100*rec:.2f}%", "yellow")
            render_metric_card("F1-score", f"{f1:.4f}", "blue")

        st.markdown("---")

        # --- Courbe ROC ---
        st.subheader("Courbe ROC")
        explain(
            "La courbe ROC montre la capacite du modele a distinguer benin vs malveillant "
            "a differents seuils. Plus la courbe est proche du coin superieur gauche, meilleur est le modele. "
            "L'<strong>AUC</strong> (aire sous la courbe) vaut 1.0 pour un modele parfait et 0.5 pour un modele aleatoire."
        )

        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC = {roc_auc:.4f})",
            line=dict(color="#3b82f6", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Modele aleatoire (AUC = 0.5)",
            line=dict(color="gray", dash="dash")
        ))
        fig_roc.update_layout(
            xaxis_title="Taux de faux positifs (FPR)",
            yaxis_title="Taux de vrais positifs (TPR / Recall)",
            template="plotly_dark", height=400, margin=dict(t=30)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    else:
        st.info(
            "Les metriques de performance ne sont disponibles que si le CSV "
            "contient une colonne **label** (0=benin, 1=malveillant). "
            "Les donnees de demonstration incluent cette colonne."
        )

    st.markdown("---")

    # --- Feature importance ---
    st.subheader("Importance des features")
    explain(
        "Ce graphique montre les 15 features les plus influentes dans les decisions du modele. "
        "L'importance est calculee par le critere de <strong>Gini</strong> : "
        "plus une feature est utilisee par le modele pour separer benin/malveillant, plus elle est importante."
    )

    if "rf_session" in models:
        model_rf = models["rf_session"]
        if hasattr(model_rf, "feature_importances_"):
            importances = model_rf.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:15]

            fig_imp = go.Figure(go.Bar(
                x=importances[sorted_idx][::-1],
                y=[session_features[i] for i in sorted_idx][::-1],
                orientation="h",
                marker_color="#3b82f6"
            ))
            fig_imp.update_layout(
                xaxis_title="Importance (Gini)",
                template="plotly_dark", height=450, margin=dict(l=300, t=30)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- Scores IF ---
    if st.session_state.get("if_scores") is not None:
        st.markdown("---")
        st.subheader("Scores d'anomalie (Isolation Forest)")

        explain(
            "L'IF attribue un <strong>score d'anomalie</strong> a chaque session. "
            "Un score negatif = la session est consideree anormale. "
            "Ce graphique superpose les distributions des scores pour les sessions benignes (bleu) "
            "et malveillantes (rouge). Si les deux histogrammes se chevauchent beaucoup, "
            "cela signifie que l'IF a du mal a les distinguer sur ces features."
        )

        if_scores = st.session_state["if_scores"]

        fig_if = go.Figure()
        if "y_true" in st.session_state:
            y_true_local = st.session_state["y_true"]
            fig_if.add_trace(go.Histogram(
                x=if_scores[y_true_local == 0], nbinsx=50, name="Sessions benignes",
                marker_color="#3b82f6", opacity=0.6
            ))
            fig_if.add_trace(go.Histogram(
                x=if_scores[y_true_local == 1], nbinsx=50, name="Sessions malveillantes",
                marker_color="#ef4444", opacity=0.6
            ))
        else:
            fig_if.add_trace(go.Histogram(
                x=if_scores, nbinsx=50, name="Toutes les sessions",
                marker_color="#3b82f6", opacity=0.7
            ))

        fig_if.add_vline(x=0, line_dash="dash", line_color="white",
                         annotation_text="Seuil IF (score=0)")
        fig_if.update_layout(
            xaxis_title="Score d'anomalie (negatif = session anormale)",
            yaxis_title="Nombre de sessions",
            barmode="overlay", template="plotly_dark", height=350,
            margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig_if, use_container_width=True)
