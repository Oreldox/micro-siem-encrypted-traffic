"""
Page : Faux negatifs — Analyse approfondie des 511 faux negatifs du Random Forest.
Donnees statiques chargees depuis data/fn_analysis.json + image t-SNE.
"""

import os
import json
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card, has_labels


def _load_fn_data():
    """Charge les donnees statiques d'analyse des faux negatifs."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "fn_analysis.json"
    )
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render(models, session_features, config):
    # =========================================================================
    # 1. Titre + en-tete contextuel
    # =========================================================================
    st.header("Analyse des faux negatifs")

    explain(
        "Un <strong>faux negatif (FN)</strong> est une session reellement malveillante "
        "que le modele a classee comme benigne. Ce sont les menaces qui passent entre les mailles du filet. "
        "Cette page analyse les <strong>511 faux negatifs</strong> du Random Forest sur le dataset "
        "CIC-Darknet2020 complet (60 859 sessions malveillantes) afin de comprendre pourquoi "
        "ces sessions echappent a la detection et comment les identifier."
    )

    # Charger les donnees statiques
    fn_data = _load_fn_data()
    if fn_data is None:
        st.error(
            "Fichier `data/fn_analysis.json` introuvable. "
            "Les donnees statiques d'analyse des faux negatifs ne sont pas disponibles."
        )
        return

    summary = fn_data["summary"]
    confidence = fn_data["confidence"]
    top_features = fn_data["top_features_diff"]
    profiles = fn_data["profiles"]

    # =========================================================================
    # 2. Metriques cles : 4 cartes
    # =========================================================================
    st.markdown("---")
    st.subheader("Vue globale")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Faux negatifs (FN)", f"{summary['fn_count']:,}", "red")
    with col2:
        render_metric_card("Vrais positifs (VP)", f"{summary['tp_count']:,}", "green")
    with col3:
        render_metric_card("Total malveillants", f"{summary['total_malicious']:,}", "blue")
    with col4:
        render_metric_card("Taux de FN", f"{summary['fn_rate']}%", "yellow")

    explain(
        f"Sur <strong>{summary['total_malicious']:,}</strong> sessions reellement malveillantes, "
        f"le modele en a correctement detecte <strong>{summary['tp_count']:,}</strong> (vrais positifs) "
        f"mais a rate <strong>{summary['fn_count']}</strong> sessions (faux negatifs). "
        f"Le taux de faux negatifs est de <strong>{summary['fn_rate']}%</strong>, "
        "ce qui correspond a un recall de 99.16%. Malgre cette excellente performance globale, "
        "comprendre ces erreurs est essentiel pour ameliorer la detection."
    )

    # =========================================================================
    # 3. Comparaison de confiance : FN vs VP
    # =========================================================================
    st.markdown("---")
    st.subheader("Confiance du modele : FN vs VP")

    explain(
        "La probabilite attribuee par le modele revele a quel point il hesite sur les faux negatifs. "
        "Un FN avec une probabilite proche de 0.5 est un cas <strong>limite</strong> — le modele hesite. "
        "Un FN avec une probabilite proche de 0 est un cas <strong>trompe</strong> — le modele est confiant dans son erreur."
    )

    col_fn, col_vp = st.columns(2)
    with col_fn:
        render_metric_card("Proba moyenne (FN)", f"{confidence['fn_mean_proba']:.3f}", "red")
        st.caption(
            f"Mediane : {confidence['fn_median_proba']:.3f} | "
            f"Min : {confidence['fn_min_proba']:.3f} | "
            f"Max : {confidence['fn_max_proba']:.3f}"
        )
    with col_vp:
        render_metric_card("Proba moyenne (VP)", f"{confidence['tp_mean_proba']:.3f}", "green")
        st.caption(
            f"Mediane : {confidence['tp_median_proba']:.3f} | "
            f"Min : {confidence['tp_min_proba']:.3f} | "
            f"Max : {confidence['tp_max_proba']:.3f}"
        )

    # Graphique a barres comparatif
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Bar(
        name="Faux negatifs (FN)",
        x=["Moyenne", "Mediane", "Min", "Max"],
        y=[
            confidence["fn_mean_proba"],
            confidence["fn_median_proba"],
            confidence["fn_min_proba"],
            confidence["fn_max_proba"]
        ],
        marker_color="#ef4444",
        opacity=0.85
    ))
    fig_conf.add_trace(go.Bar(
        name="Vrais positifs (VP)",
        x=["Moyenne", "Mediane", "Min", "Max"],
        y=[
            confidence["tp_mean_proba"],
            confidence["tp_median_proba"],
            confidence["tp_min_proba"],
            confidence["tp_max_proba"]
        ],
        marker_color="#10b981",
        opacity=0.85
    ))
    fig_conf.add_hline(
        y=0.5, line_dash="dash", line_color="white",
        annotation_text="Seuil de detection (0.5)",
        annotation_position="top left"
    )
    fig_conf.update_layout(
        barmode="group",
        yaxis_title="Probabilite P(malveillant)",
        xaxis_title="Statistique",
        template="plotly_dark",
        height=400,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    explain(
        f"Les faux negatifs ont une probabilite moyenne de <strong>{confidence['fn_mean_proba']:.3f}</strong>, "
        f"bien en dessous du seuil de 0.5, tandis que les vrais positifs culminent a "
        f"<strong>{confidence['tp_mean_proba']:.3f}</strong>. "
        "La majorite des FN se situent entre 0.3 et 0.5 — ce sont des cas limites ou le modele "
        "hesite mais penche du mauvais cote. Un abaissement du seuil pourrait recuperer une partie "
        "de ces sessions, au prix de quelques faux positifs supplementaires."
    )

    # =========================================================================
    # 4. Top 10 features discriminantes
    # =========================================================================
    st.markdown("---")
    st.subheader("Top 10 features discriminantes (FN vs VP)")

    explain(
        "Ces features presentent les plus grandes differences entre les faux negatifs et les vrais positifs. "
        "Un ecart positif signifie que les FN ont des valeurs <strong>plus elevees</strong> que les VP "
        "pour cette feature. Un ecart negatif signifie des valeurs <strong>plus faibles</strong>. "
        "Ces differences expliquent pourquoi le modele confond ces sessions malveillantes avec du trafic benin."
    )

    # Trier par valeur absolue decroissante pour le graphique
    features_sorted = sorted(top_features, key=lambda x: abs(x["diff_pct"]))

    feature_names = [f["feature"] for f in features_sorted]
    diff_values = [f["diff_pct"] for f in features_sorted]
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in diff_values]

    fig_features = go.Figure(go.Bar(
        x=diff_values,
        y=feature_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in diff_values],
        textposition="outside"
    ))
    fig_features.update_layout(
        xaxis_title="Ecart FN vs VP (%)",
        yaxis_title="",
        template="plotly_dark",
        height=500,
        margin=dict(l=350, t=30, r=80),
        xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1)
    )
    st.plotly_chart(fig_features, use_container_width=True)

    # Tableau d'interpretation
    st.markdown("**Interpretation des ecarts :**")

    table_header = "| Feature | Ecart (%) | Interpretation |"
    table_sep = "|---|---|---|"
    table_rows = []
    for f in top_features:
        table_rows.append(
            f"| `{f['feature']}` | {f['diff_pct']:+.1f}% | {f['interpretation']} |"
        )

    st.markdown("\n".join([table_header, table_sep] + table_rows))

    explain(
        "Les faux negatifs se distinguent par des <strong>intervalles d'arrivee plus longs</strong>, "
        "des <strong>paquets plus gros et plus variables</strong>, et des <strong>changements TCP plus dynamiques</strong>. "
        "Ce profil ressemble davantage a du trafic humain qu'a du trafic automatise, "
        "ce qui explique pourquoi le modele les confond avec du trafic benin."
    )

    # =========================================================================
    # 5. Profils comparatifs : VP / FN / Benin
    # =========================================================================
    st.markdown("---")
    st.subheader("Profils comportementaux")

    explain(
        "Trois profils types emergent de l'analyse. Le profil des faux negatifs se situe "
        "<strong>entre</strong> celui des vrais positifs et celui du trafic benin, "
        "ce qui explique la difficulte du modele a les classifier correctement."
    )

    col_vp, col_fn, col_benin = st.columns(3)

    with col_vp:
        tp_profile = profiles["tp"]
        render_metric_card(tp_profile["label"], f"P = {tp_profile['proba']:.3f}", "green")
        st.markdown(f"**Description :**")
        st.markdown(tp_profile["description"])

    with col_fn:
        fn_profile = profiles["fn"]
        render_metric_card(fn_profile["label"], f"P = {fn_profile['proba']:.3f}", "red")
        st.markdown(f"**Description :**")
        st.markdown(fn_profile["description"])

    with col_benin:
        benign_profile = profiles["benign"]
        render_metric_card(benign_profile["label"], f"P = {benign_profile['proba']:.3f}", "blue")
        st.markdown(f"**Description :**")
        st.markdown(benign_profile["description"])

    explain(
        "Le profil <strong>FN</strong> (malware rate) ressemble fortement au profil <strong>benin</strong> : "
        "gros paquets, taille variable, longs intervalles. Ces malwares imitent le comportement humain, "
        "ce qui les rend difficiles a distinguer du trafic legitime par les seules metadonnees de session. "
        "Le profil <strong>VP</strong> (malware detecte) est bien distinct : paquets petits et uniformes, "
        "comportement automatise typique."
    )

    # =========================================================================
    # 6. Visualisation t-SNE
    # =========================================================================
    st.markdown("---")
    st.subheader("Visualisation t-SNE des faux negatifs")

    explain(
        "La projection <strong>t-SNE</strong> reduit les 27 dimensions des features en 2D "
        "tout en preservant les distances locales. Cette visualisation montre ou se situent "
        "les faux negatifs dans l'espace des features par rapport aux vrais positifs et au trafic benin."
    )

    image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "images", "tsne_faux_negatifs.png"
    )

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

        explain(
            "<strong>Guide de lecture :</strong> Chaque point represente une session reseau. "
            "Les couleurs distinguent les vrais positifs (malware detecte), les faux negatifs "
            "(malware rate) et le trafic benin. Observez la position des faux negatifs : "
            "s'ils se trouvent <strong>pres des sessions benignes</strong>, cela confirme "
            "que leur profil reseau imite le trafic legitime. S'ils forment un <strong>cluster separe</strong>, "
            "une feature supplementaire pourrait aider a les distinguer."
        )
    else:
        st.warning(
            "Image `data/images/tsne_faux_negatifs.png` introuvable. "
            "La visualisation t-SNE ne peut pas etre affichee."
        )

    # =========================================================================
    # 7. Section dynamique : FN sur les donnees actuelles
    # =========================================================================
    st.markdown("---")
    st.subheader("Faux negatifs sur les donnees actuelles")

    if "probas" in st.session_state and "y_true" in st.session_state:
        probas = st.session_state["probas"]
        y_true = st.session_state["y_true"]
        threshold = config["threshold"]

        preds = (probas >= threshold).astype(int)

        # Calcul des FN actuels
        fn_mask = (y_true == 1) & (preds == 0)
        tp_mask = (y_true == 1) & (preds == 1)

        n_fn_actual = int(fn_mask.sum())
        n_tp_actual = int(tp_mask.sum())
        n_malicious_actual = int((y_true == 1).sum())
        fn_rate_actual = (100 * n_fn_actual / max(n_malicious_actual, 1))

        explain(
            f"Les donnees actuellement chargees contiennent des <strong>labels</strong> (verite terrain). "
            f"Voici les faux negatifs calcules avec le seuil actuel de <strong>{threshold}</strong>."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("FN (donnees actuelles)", f"{n_fn_actual:,}", "red")
        with col2:
            render_metric_card("VP (donnees actuelles)", f"{n_tp_actual:,}", "green")
        with col3:
            render_metric_card("Total malveillants", f"{n_malicious_actual:,}", "blue")
        with col4:
            render_metric_card("Taux de FN", f"{fn_rate_actual:.2f}%", "yellow")

        # Comparaison avec les donnees de reference
        st.markdown("**Comparaison avec l'analyse de reference (dataset complet) :**")

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Reference (dataset complet)",
            x=["Faux negatifs", "Vrais positifs", "Taux FN (%)"],
            y=[summary["fn_count"], summary["tp_count"], summary["fn_rate"]],
            marker_color="#3b82f6",
            opacity=0.7
        ))
        fig_compare.add_trace(go.Bar(
            name=f"Donnees actuelles (seuil={threshold})",
            x=["Faux negatifs", "Vrais positifs", "Taux FN (%)"],
            y=[n_fn_actual, n_tp_actual, fn_rate_actual],
            marker_color="#f59e0b",
            opacity=0.85
        ))
        fig_compare.update_layout(
            barmode="group",
            yaxis_title="Valeur",
            template="plotly_dark",
            height=400,
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Distribution des probabilites des FN actuels
        if n_fn_actual > 0:
            fn_probas = probas[fn_mask]
            tp_probas = probas[tp_mask]

            st.markdown("**Distribution des probabilites des faux negatifs actuels :**")

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=fn_probas, nbinsx=30,
                name=f"Faux negatifs ({n_fn_actual})",
                marker_color="#ef4444", opacity=0.7
            ))
            if n_tp_actual > 0:
                fig_dist.add_trace(go.Histogram(
                    x=tp_probas, nbinsx=50,
                    name=f"Vrais positifs ({n_tp_actual})",
                    marker_color="#10b981", opacity=0.4
                ))
            fig_dist.add_vline(
                x=threshold, line_dash="dash", line_color="white",
                annotation_text=f"Seuil = {threshold}"
            )
            fig_dist.update_layout(
                barmode="overlay",
                xaxis_title="Probabilite P(malveillant)",
                yaxis_title="Nombre de sessions",
                template="plotly_dark",
                height=350,
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            fn_mean = float(fn_probas.mean())
            fn_median = float(np.median(fn_probas))
            explain(
                f"Les <strong>{n_fn_actual}</strong> faux negatifs actuels ont une probabilite "
                f"moyenne de <strong>{fn_mean:.3f}</strong> (mediane : {fn_median:.3f}). "
                f"Tous sont en dessous du seuil de {threshold}. "
                "Un abaissement du seuil pourrait recuperer les cas limites, "
                "mais augmenterait le nombre de faux positifs."
            )
        else:
            st.success(
                "Aucun faux negatif detecte sur les donnees actuelles avec le seuil de "
                f"{threshold}. Toutes les sessions malveillantes ont ete correctement identifiees."
            )
    else:
        st.info(
            "Aucune donnee labelisee chargee. Pour voir les faux negatifs sur vos donnees, "
            "chargez les donnees de demonstration (qui incluent les labels de verite terrain) "
            "depuis la page **Vue d'ensemble**."
        )

        explain(
            "L'analyse dynamique necessite des donnees avec des <strong>labels</strong> (colonne `label`) "
            "pour pouvoir comparer les predictions du modele a la verite terrain. "
            "Sans labels, seule l'analyse statique de reference (ci-dessus) est disponible."
        )
