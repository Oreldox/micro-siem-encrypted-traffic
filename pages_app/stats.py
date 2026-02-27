"""
Page 6 : Statistiques — Feature importance, scores IF, distribution,
matrice de confusion et ROC (si labels disponibles).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card, has_labels


def render(models, session_features, config):
    st.header("Statistiques du modele")

    explain(
        "Cette page presente les statistiques de performance et de comportement du modele. "
        "L'<strong>importance des features</strong> montre quelles caracteristiques du trafic "
        "influencent le plus les decisions du modele. "
        "La <strong>distribution des scores</strong> indique la confiance globale du modele. "
        "Si des labels sont disponibles, des metriques de precision (matrice de confusion, ROC) sont aussi affichees."
    )

    from src.ui_components import require_data
    if not require_data("Feature importance, scores IF, metriques de performance."):
        return

    probas = st.session_state["probas"]
    preds = st.session_state["preds"]

    # Resume rapide
    n_total = len(probas)
    n_confident = int(((probas < 0.2) | (probas > 0.8)).sum())
    n_uncertain = int(((probas >= 0.3) & (probas <= 0.7)).sum())
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Sessions analysees", f"{n_total:,}", "blue")
    with col2:
        render_metric_card("Predictions confiantes", f"{n_confident:,}", "green")
    with col3:
        render_metric_card("Zone d'incertitude", f"{n_uncertain:,}", "yellow")

    if n_uncertain > n_total * 0.2:
        st.caption(
            f"{n_uncertain:,} sessions ({100*n_uncertain/n_total:.0f}%) sont dans la zone grise (P entre 0.3 et 0.7). "
            "Envisagez le Mode cascade ou ajustez le seuil."
        )

    st.markdown("---")

    if has_labels():
        _render_confusion_matrix(preds, probas)
        st.markdown("---")

    if not has_labels():
        st.info(
            "Les donnees importees ne contiennent pas de labels (verite terrain). "
            "La matrice de confusion et la courbe ROC ne sont pas disponibles — "
            "c'est normal pour du trafic reel."
        )

    # Toujours afficher feature importance + distribution + IF
    _render_feature_importance(models, session_features)
    _render_model_comparison(models, probas)
    _render_probability_distribution(probas, config)
    _render_if_scores(config)


def _render_confusion_matrix(preds, probas):
    """Matrice de confusion + metriques + ROC (necessite labels)."""
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

    # Courbe ROC
    st.subheader("Courbe ROC")
    explain(
        "La courbe ROC montre la capacite du modele a distinguer benin vs malveillant "
        "a differents seuils. Plus la courbe est proche du coin superieur gauche, meilleur est le modele. "
        "L'<strong>AUC</strong> (aire sous la courbe) vaut 1.0 pour un modele parfait et 0.5 pour un modele aleatoire."
    )

    fpr, tpr, thresholds_roc = roc_curve(y_true, probas)
    roc_auc = auc(fpr, tpr)

    # Youden's J = TPR - FPR (seuil optimal = max de J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds_roc[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]
    best_j = j_scores[best_idx]

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC = {roc_auc:.4f})",
        line=dict(color="#3b82f6", width=2.5)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Modele aleatoire (AUC = 0.5)",
        line=dict(color="gray", dash="dash")
    ))
    # Point optimal de Youden
    fig_roc.add_trace(go.Scatter(
        x=[best_fpr], y=[best_tpr], mode="markers+text",
        name=f"Youden (J = {best_j:.4f})",
        marker=dict(color="#f59e0b", size=12, symbol="star"),
        text=[f"Seuil = {best_threshold:.4f}"],
        textposition="bottom right",
        textfont=dict(color="#f59e0b", size=11),
    ))
    fig_roc.update_layout(
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (TPR / Recall)",
        template="plotly_dark", height=400, margin=dict(t=30)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Metriques Youden's J
    col_j1, col_j2, col_j3, col_j4 = st.columns(4)
    with col_j1:
        render_metric_card("Youden's J max", f"{best_j:.4f}", "yellow")
    with col_j2:
        render_metric_card("Seuil optimal", f"{best_threshold:.4f}", "yellow")
    with col_j3:
        render_metric_card("TPR au seuil", f"{best_tpr:.4f}", "green")
    with col_j4:
        render_metric_card("FPR au seuil", f"{best_fpr:.4f}", "red")

    explain(
        "L'indice de <strong>Youden (J = TPR - FPR)</strong> identifie le seuil qui maximise "
        "la distance entre la courbe ROC et la diagonale aleatoire. "
        f"Le seuil optimal est <strong>{best_threshold:.4f}</strong> (J = {best_j:.4f}), "
        f"avec un TPR de {best_tpr:.4f} et un FPR de {best_fpr:.4f}."
    )


def _render_feature_importance(models, session_features):
    """Top 15 features par importance Gini."""
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

    st.markdown("---")


def _render_probability_distribution(probas, config):
    """Distribution des scores RF (utile quand pas de labels)."""
    st.subheader("Distribution des scores de classification")
    explain(
        "Distribution des probabilites P(malveillant) attribuees par le Random Forest. "
        "Un modele confiant produit une distribution bimodale (pics pres de 0 et 1). "
        "Les sessions dans la zone grise (0.3-0.7) meritent une attention particuliere."
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=probas, nbinsx=50, name="Toutes les sessions",
        marker_color="#3b82f6", opacity=0.7
    ))
    fig.add_vline(x=config["threshold"], line_dash="dash", line_color="red",
                  annotation_text=f"Seuil = {config['threshold']}")
    fig.update_layout(
        xaxis_title="Probabilite de malveillance",
        yaxis_title="Nombre de sessions",
        template="plotly_dark", height=350, margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


def _render_if_scores(config):
    """Scores d'anomalie IF (si active)."""
    if st.session_state.get("if_scores") is None:
        return

    st.subheader("Scores d'anomalie (Isolation Forest)")

    explain(
        "L'IF attribue un <strong>score d'anomalie</strong> a chaque session. "
        "Un score negatif = la session est consideree anormale. "
        "Les sessions a gauche du seuil (score < 0) sont des anomalies."
    )

    if_scores = st.session_state["if_scores"]

    fig_if = go.Figure()
    if has_labels():
        y_true = st.session_state["y_true"]
        fig_if.add_trace(go.Histogram(
            x=if_scores[y_true == 0], nbinsx=50, name="Sessions benignes",
            marker_color="#3b82f6", opacity=0.6
        ))
        fig_if.add_trace(go.Histogram(
            x=if_scores[y_true == 1], nbinsx=50, name="Sessions malveillantes",
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


def _render_model_comparison(models, probas_rf):
    """Compare les predictions RF vs XGBoost (si disponible)."""
    probas_xgb = st.session_state.get("probas_xgb")
    if probas_xgb is None:
        return

    st.subheader("Accord inter-modeles (RF vs XGBoost)")
    explain(
        "Ce graphique compare les probabilites du <strong>Random Forest</strong> (modele principal) "
        "et du <strong>XGBoost</strong>. Les points sur la diagonale = les deux modeles sont d'accord. "
        "Les points loin de la diagonale = desaccord → sessions a investiguer."
    )

    # Accord global
    agree_05 = ((probas_rf >= 0.5) == (probas_xgb >= 0.5)).mean()
    correlation = np.corrcoef(probas_rf, probas_xgb)[0, 1]

    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Accord sur le verdict", f"{agree_05:.1%}", "green" if agree_05 > 0.95 else "yellow")
    with col2:
        render_metric_card("Correlation des probas", f"{correlation:.4f}", "green" if correlation > 0.95 else "yellow")

    # Scatter plot
    fig = go.Figure()

    # Limiter a 5000 points pour la perf
    n = len(probas_rf)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        rf_sub = probas_rf[idx]
        xgb_sub = probas_xgb[idx]
    else:
        rf_sub = probas_rf
        xgb_sub = probas_xgb

    fig.add_trace(go.Scattergl(
        x=rf_sub, y=xgb_sub,
        mode="markers",
        marker=dict(size=3, color="#3b82f6", opacity=0.4),
        name="Sessions"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash"), name="Accord parfait"
    ))
    fig.update_layout(
        xaxis_title="Probabilite RF",
        yaxis_title="Probabilite XGBoost",
        template="plotly_dark", height=400,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sessions en desaccord
    disagree_mask = (probas_rf >= 0.5) != (probas_xgb >= 0.5)
    n_disagree = disagree_mask.sum()
    if n_disagree > 0:
        st.caption(
            f"**{n_disagree}** sessions ({100*n_disagree/n:.1f}%) ou les modeles sont en desaccord. "
            "Ces sessions meritent une investigation manuelle."
        )

    st.markdown("---")
