"""
Page Comparer les datasets — Tableau recapitulatif des performances
sur chaque dataset teste pendant la session.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card, render_pedagogy


TRAINING_REF = {
    "accuracy": 99.50,
    "precision": 99.84,
    "recall": 99.16,
    "f1": 0.9950,
}


def render():
    st.header("Comparer les datasets")

    explain(
        "Cette page recapitule les performances du modele sur chaque dataset teste "
        "pendant cette session. Elle permet de comparer la generalisation du modele "
        "et de comprendre sur quel type de donnees il performe le mieux."
    )

    history = st.session_state.get("comparison_history", {})

    if not history:
        st.info(
            "Aucun dataset teste pour l'instant. "
            "Rendez-vous sur la page **Tester le modele** pour analyser un dataset. "
            "Les resultats apparaitront automatiquement ici."
        )
        return

    # === Tableau recapitulatif ===
    st.subheader("Tableau comparatif")

    rows = []
    for name, metrics in history.items():
        rows.append({
            "Dataset": name[:60],
            "Sessions": metrics.get("n_sessions", 0),
            "Features": f"{metrics.get('features', 27)}/27",
            "Seuil": metrics.get("threshold", 0.5),
            "Accuracy": f"{metrics['accuracy']:.2f}%",
            "Precision": f"{metrics['precision']:.2f}%",
            "Recall": f"{metrics['recall']:.2f}%",
            "F1-score": f"{metrics['f1']:.4f}",
            "Alertes": metrics.get("n_alerts", 0),
        })

    # Ajouter la reference d'entrainement
    rows.append({
        "Dataset": "Entrainement (reference)",
        "Sessions": 122_132,
        "Features": "27/27",
        "Seuil": 0.5,
        "Accuracy": f"{TRAINING_REF['accuracy']:.2f}%",
        "Precision": f"{TRAINING_REF['precision']:.2f}%",
        "Recall": f"{TRAINING_REF['recall']:.2f}%",
        "F1-score": f"{TRAINING_REF['f1']:.4f}",
        "Alertes": "-",
    })

    df_comp = pd.DataFrame(rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # === Graphique comparatif F1 ===
    st.markdown("---")
    st.subheader("F1-score par dataset")

    dataset_names = []
    f1_scores = []
    colors = []

    # Reference
    dataset_names.append("Entrainement")
    f1_scores.append(TRAINING_REF["f1"])
    colors.append("#3b82f6")

    for name, metrics in history.items():
        short_name = name[:40]
        dataset_names.append(short_name)
        f1 = metrics["f1"]
        f1_scores.append(f1)
        if f1 >= 0.95:
            colors.append("#10b981")
        elif f1 >= 0.7:
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")

    fig = go.Figure(go.Bar(
        x=dataset_names,
        y=f1_scores,
        marker_color=colors,
        text=[f"{f:.4f}" for f in f1_scores],
        textposition="outside",
    ))
    fig.add_hline(y=TRAINING_REF["f1"], line_dash="dash", line_color="#3b82f6",
                  annotation_text=f"Reference = {TRAINING_REF['f1']:.4f}")
    fig.update_layout(
        yaxis_title="F1-score",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        height=400,
        margin=dict(t=30, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Graphique radar multi-datasets ===
    if len(history) >= 2:
        st.markdown("---")
        st.subheader("Comparaison radar")

        metrics_names = ["Accuracy", "Precision", "Recall", "F1 x100"]
        fig_radar = go.Figure()

        # Reference
        ref_vals = [TRAINING_REF["accuracy"], TRAINING_REF["precision"],
                    TRAINING_REF["recall"], TRAINING_REF["f1"] * 100]
        ref_vals_closed = ref_vals + [ref_vals[0]]
        metrics_closed = metrics_names + [metrics_names[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=ref_vals_closed, theta=metrics_closed,
            fill="toself", name="Entrainement",
            fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="#3b82f6", width=2, dash="dash")
        ))

        palette = ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
        for i, (name, metrics) in enumerate(history.items()):
            vals = [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"] * 100,
            ]
            vals_closed = vals + [vals[0]]
            color = palette[i % len(palette)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=metrics_closed,
                fill="toself", name=name[:30],
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
                line=dict(color=color, width=2)
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"),
                angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)"
            ),
            template="plotly_dark", height=450,
            margin=dict(t=30, b=30),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # === Interpretation ===
    st.markdown("---")
    st.subheader("Interpretation")

    # Trouver le meilleur et le pire
    if history:
        best_name = max(history, key=lambda k: history[k]["f1"])
        worst_name = min(history, key=lambda k: history[k]["f1"])
        best_f1 = history[best_name]["f1"]
        worst_f1 = history[worst_name]["f1"]

        if best_f1 > 0.9:
            st.success(
                f"**Meilleur resultat** : {best_name[:50]} (F1 = {best_f1:.4f}). "
                "Le modele generalise bien sur ce dataset."
            )
        if worst_f1 < 0.5 and len(history) > 1:
            st.error(
                f"**Moins bon resultat** : {worst_name[:50]} (F1 = {worst_f1:.4f}). "
                "Le modele ne generalise pas sur ce dataset."
            )

    render_pedagogy(
        "<strong>Que retenir de cette comparaison ?</strong><br><br>"
        "1. Le modele performe excellemment sur des donnees <strong>similaires a l'entrainement</strong> "
        "(CIC-Darknet2020).<br>"
        "2. La performance se degrade sur des datasets <strong>extraits differemment</strong> "
        "ou contenant des <strong>familles de malware inconnues</strong>.<br>"
        "3. C'est un comportement <strong>normal et attendu</strong> en Machine Learning supervise. "
        "Un modele n'est pas une verite universelle — c'est un outil specialise.<br>"
        "4. Pour ameliorer la generalisation, il faudrait <strong>re-entrainer sur un corpus plus large</strong> "
        "combinant plusieurs sources de donnees."
    )
