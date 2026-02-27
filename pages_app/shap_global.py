"""
Page SHAP Global — Analyse SHAP globale : beeswarm, ranking, dependance,
comparaison SHAP vs Gini, comparaison inter-algorithmes, SHAP dynamique.
"""

import os
import json
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SHAP_STATS_PATH = os.path.join(DATA_DIR, "shap_stats.json")


def _load_shap_stats():
    """Charge les statistiques SHAP pre-calculees depuis shap_stats.json."""
    if not os.path.exists(SHAP_STATS_PATH):
        return None
    with open(SHAP_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render(models, session_features, config):
    # =========================================================================
    # 1. TITRE + INTRODUCTION
    # =========================================================================
    st.header("SHAP - Expliquer les decisions du modele")

    explain(
        "Les <strong>valeurs SHAP</strong> (SHapley Additive exPlanations) sont issues de la theorie des jeux cooperatifs. "
        "Pour chaque prediction, SHAP attribue a chaque feature une <strong>contribution</strong> "
        "qui mesure son impact sur la decision du modele. "
        "Cette page presente l'analyse SHAP <strong>globale</strong> : "
        "quelles features influencent le plus le modele sur l'ensemble du dataset, "
        "comment elles interagissent, et comment le classement SHAP se compare a l'importance Gini classique."
    )

    # Charger les stats pre-calculees
    stats = _load_shap_stats()
    if stats is None:
        st.error("Fichier `data/shap_stats.json` introuvable. L'analyse SHAP globale pre-calculee n'est pas disponible.")
        return

    shap_config = stats.get("config", {})
    ranking = stats.get("ranking", [])
    shap_vs_gini = stats.get("shap_vs_gini", {})

    # =========================================================================
    # 2. CONFIGURATION DE L'ANALYSE
    # =========================================================================
    st.markdown("---")
    st.subheader("Configuration de l'analyse SHAP")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Modele", shap_config.get("model", "XGBoost"), "blue")
    with col2:
        render_metric_card("Methode", shap_config.get("method", "TreeExplainer"), "blue")
    with col3:
        render_metric_card("Echantillon", f"{shap_config.get('sample', 5000):,}", "green")
    with col4:
        render_metric_card("Temps de calcul", f"{shap_config.get('compute_time', 2.2)}s", "green")

    explain(
        "L'analyse SHAP a ete realisee avec <strong>TreeExplainer</strong>, "
        "un algorithme exact et rapide pour les modeles a base d'arbres (XGBoost, Random Forest). "
        "Contrairement aux approximations (KernelSHAP), TreeExplainer calcule les valeurs SHAP exactes "
        "en exploitant la structure des arbres de decision."
    )

    # =========================================================================
    # 3. SHAP SUMMARY (BEESWARM)
    # =========================================================================
    st.markdown("---")
    st.subheader("SHAP Summary Plot (Beeswarm)")

    shap_summary_path = os.path.join(IMAGES_DIR, "shap_summary.png")
    if os.path.exists(shap_summary_path):
        st.image(shap_summary_path, use_container_width=True)
    else:
        st.warning("Image `data/images/shap_summary.png` introuvable.")

    explain(
        "<strong>Guide de lecture du beeswarm plot :</strong><br>"
        "- <strong>Axe Y</strong> : les features, classees de la plus influente (en haut) a la moins influente (en bas).<br>"
        "- <strong>Axe X</strong> : la valeur SHAP — a droite, la feature pousse vers la classe <em>malveillant</em> ; "
        "a gauche, vers <em>benin</em>.<br>"
        "- <strong>Couleur</strong> : la valeur reelle de la feature — "
        "<span style='color:#ef4444'>rouge</span> = valeur elevee, "
        "<span style='color:#3b82f6'>bleu</span> = valeur basse.<br>"
        "- Chaque point represente une session du dataset. La dispersion horizontale montre "
        "la variabilite de l'impact d'une feature selon les sessions."
    )

    # =========================================================================
    # 4. RANKING SHAP — BAR CHART
    # =========================================================================
    st.markdown("---")
    st.subheader("Classement des features par importance SHAP")

    if ranking:
        feature_names = [r["feature"] for r in ranking][::-1]
        mean_shap_values = [r["mean_shap"] for r in ranking][::-1]

        fig_ranking = go.Figure(go.Bar(
            x=mean_shap_values,
            y=feature_names,
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:.4f}" for v in mean_shap_values],
            textposition="outside",
            textfont=dict(size=11)
        ))
        fig_ranking.update_layout(
            xaxis_title="mean |SHAP value|",
            yaxis_title="",
            template="plotly_dark",
            height=450,
            margin=dict(l=350, t=30, r=80),
            xaxis=dict(range=[0, max(mean_shap_values) * 1.15])
        )
        st.plotly_chart(fig_ranking, use_container_width=True)

        explain(
            "Ce graphique classe les 10 features les plus influentes selon la <strong>moyenne des valeurs |SHAP|</strong> "
            "sur l'ensemble de l'echantillon. Contrairement a l'importance Gini (qui mesure la purete des splits), "
            "SHAP quantifie la <strong>contribution reelle</strong> de chaque feature aux predictions individuelles.<br><br>"
            f"La feature la plus importante est <strong>{ranking[0]['feature']}</strong> "
            f"avec un mean |SHAP| de <strong>{ranking[0]['mean_shap']:.4f}</strong>, "
            f"suivie de <strong>{ranking[1]['feature']}</strong> ({ranking[1]['mean_shap']:.4f})."
        )
    else:
        st.warning("Aucune donnee de ranking SHAP disponible.")

    # =========================================================================
    # 5. SHAP vs GINI
    # =========================================================================
    st.markdown("---")
    st.subheader("Comparaison SHAP vs Gini (Top 10)")

    overlap = shap_vs_gini.get("overlap_top10", 0)
    shap_top10 = shap_vs_gini.get("shap_top10", [])
    gini_top10 = shap_vs_gini.get("gini_top10", [])

    # Metrique de recouvrement
    col_metric, col_spacer = st.columns([1, 2])
    with col_metric:
        overlap_pct = (overlap / 10 * 100) if overlap else 0
        render_metric_card(
            "Features en commun",
            f"{overlap}/10 ({overlap_pct:.0f}%)",
            "green" if overlap >= 7 else ("yellow" if overlap >= 5 else "red")
        )

    explain(
        f"<strong>{overlap} features sur 10</strong> sont presentes dans les deux classements (top 10 SHAP et top 10 Gini). "
        "Un recouvrement eleve indique que les deux methodes convergent sur les features les plus discriminantes. "
        "Les differences de classement revelent les features qui <em>contribuent fortement aux predictions individuelles</em> (SHAP) "
        "vs celles qui <em>separent globalement les classes</em> (Gini)."
    )

    # Tableau comparatif cote a cote
    if shap_top10 and gini_top10:
        max_len = max(len(shap_top10), len(gini_top10))
        shap_set = set(shap_top10)
        gini_set = set(gini_top10)

        table_data = []
        for i in range(max_len):
            shap_feat = shap_top10[i] if i < len(shap_top10) else ""
            gini_feat = gini_top10[i] if i < len(gini_top10) else ""
            # Marquer les features presentes dans les deux classements
            shap_marker = " *" if shap_feat and shap_feat in gini_set else ""
            gini_marker = " *" if gini_feat and gini_feat in shap_set else ""
            table_data.append({
                "Rang": i + 1,
                "SHAP Top 10": f"{shap_feat}{shap_marker}",
                "Gini Top 10": f"{gini_feat}{gini_marker}"
            })

        import pandas as pd
        df_comparison = pd.DataFrame(table_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        st.caption("Les features marquees d'un * sont presentes dans les deux classements.")

        # Identifier les features uniques a chaque methode
        only_shap = shap_set - gini_set
        only_gini = gini_set - shap_set
        if only_shap:
            st.markdown(f"**Uniquement dans le top 10 SHAP** : {', '.join(only_shap)}")
        if only_gini:
            st.markdown(f"**Uniquement dans le top 10 Gini** : {', '.join(only_gini)}")

    # =========================================================================
    # 6. DEPENDENCE PLOTS
    # =========================================================================
    st.markdown("---")
    st.subheader("SHAP Dependence Plots (Top 3 features)")

    shap_dep_path = os.path.join(IMAGES_DIR, "shap_dependence_top3.png")
    if os.path.exists(shap_dep_path):
        st.image(shap_dep_path, use_container_width=True)
    else:
        st.warning("Image `data/images/shap_dependence_top3.png` introuvable.")

    explain(
        "<strong>Les dependence plots</strong> montrent la relation entre la valeur d'une feature (axe X) "
        "et son impact SHAP (axe Y) pour chaque session.<br><br>"
        "- <strong>max_Change_values_of_TCP_windows_length_per_session</strong> : "
        "un nombre eleve de changements de fenetre TCP augmente fortement la probabilite de malveillance. "
        "Les outils automatises (C2, bots) ont des comportements TCP previsibles et repetitifs.<br>"
        "- <strong>IPratio_ratio</strong> : "
        "un ratio desequilibre entre paquets aller/retour (exfiltration, scan) "
        "pousse le modele vers la classification malveillante.<br>"
        "- <strong>median_ttl_backward_traffic</strong> : "
        "un TTL median anormal dans le trafic retour peut indiquer du spoofing IP, "
        "un proxy ou un serveur C2 situe dans une infrastructure inhabituelle.<br><br>"
        "La couleur represente une feature d'interaction selectionnee automatiquement par SHAP, "
        "revelant les correlations entre features."
    )

    # =========================================================================
    # 7. COMPARAISON INTER-ALGORITHMES
    # =========================================================================
    st.markdown("---")
    st.subheader("Comparaison des importances : RF vs XGBoost vs MLP")

    comp_path = os.path.join(IMAGES_DIR, "comparaison_importances.png")
    if os.path.exists(comp_path):
        st.image(comp_path, use_container_width=True)
    else:
        st.warning("Image `data/images/comparaison_importances.png` introuvable.")

    explain(
        "Ce graphique compare l'importance des features selon trois algorithmes differents : "
        "<strong>Random Forest</strong> (Gini), <strong>XGBoost</strong> (gain), et <strong>MLP</strong> (permutation).<br><br>"
        "- Si les trois modeles s'accordent sur les features les plus importantes, "
        "cela renforce la <strong>robustesse</strong> de l'analyse : les features discriminantes ne dependent pas "
        "d'un algorithme particulier.<br>"
        "- Les differences revelent les biais propres a chaque methode : "
        "les arbres (RF, XGBoost) exploitent les seuils de decision, "
        "tandis que le MLP capte des relations non-lineaires complexes.<br>"
        "- Les features systematiquement en tete (TCP window, IP ratio, TTL) sont les plus fiables "
        "pour distinguer le trafic malveillant du trafic benin, independamment de l'algorithme utilise."
    )

    # =========================================================================
    # 8. SHAP DYNAMIQUE
    # =========================================================================
    st.markdown("---")
    st.subheader("SHAP dynamique (sur vos donnees)")

    explain(
        "Si des donnees sont chargees et que le modele XGBoost est disponible, "
        "cette section calcule les valeurs SHAP en temps reel sur un sous-echantillon "
        "de vos donnees et affiche un <strong>beeswarm interactif</strong>."
    )

    # Verifier les prerequis
    has_data = "X" in st.session_state and "data" in st.session_state
    has_xgboost = "xgboost" in models

    if not has_data:
        st.info(
            "Aucune donnee chargee. Chargez les donnees de demonstration ou importez vos donnees "
            "depuis la page Vue d'ensemble pour activer le SHAP dynamique."
        )
        return

    if not has_xgboost:
        st.warning(
            "Le modele XGBoost n'est pas disponible. "
            "Le SHAP dynamique necessite le modele XGBoost pour le calcul des valeurs SHAP."
        )
        return

    X = st.session_state["X"]
    n_total = len(X)

    # Sous-echantillonnage pour la performance
    max_sample = 500
    if n_total > max_sample:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_total, max_sample, replace=False)
        X_sample = X[sample_idx]
        st.caption(f"Sous-echantillon : {max_sample} sessions sur {n_total:,} (pour la performance du calcul SHAP).")
    else:
        X_sample = X
        sample_idx = np.arange(n_total)
        st.caption(f"Calcul SHAP sur les {n_total:,} sessions.")

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.models import load_shap_explainer

        with st.spinner("Calcul des valeurs SHAP (TreeExplainer sur XGBoost)..."):
            explainer = load_shap_explainer(models["xgboost"])
            shap_values = explainer.shap_values(X_sample)

            # XGBoost retourne directement un array (pas une liste comme sklearn RF)
            if isinstance(shap_values, list):
                sv = shap_values[1]  # classe 1 (malveillant)
            else:
                sv = shap_values

            # Beeswarm interactif avec matplotlib/SHAP
            explanation = shap.Explanation(
                values=sv,
                base_values=(
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                ),
                data=X_sample,
                feature_names=list(session_features)
            )

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.beeswarm(explanation, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Statistiques du SHAP dynamique
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:5]

        col_dyn1, col_dyn2, col_dyn3 = st.columns(3)
        with col_dyn1:
            render_metric_card("Sessions analysees", f"{len(X_sample):,}", "blue")
        with col_dyn2:
            render_metric_card("Features", f"{len(session_features)}", "blue")
        with col_dyn3:
            top_feat_name = session_features[top_idx[0]] if len(session_features) > top_idx[0] else "N/A"
            render_metric_card("Feature #1", top_feat_name, "green")

        st.markdown("**Top 5 features (SHAP dynamique) :**")
        for rank, fi in enumerate(top_idx, 1):
            feat_name = session_features[fi] if fi < len(session_features) else f"feature_{fi}"
            st.markdown(f"{rank}. **{feat_name}** — mean |SHAP| = {mean_abs_shap[fi]:.4f}")

        explain(
            "Ce beeswarm a ete calcule dynamiquement sur vos donnees. "
            "Comparez-le avec le beeswarm pre-calcule ci-dessus pour verifier "
            "si les memes features dominent. Une coherence entre les deux renforce "
            "la fiabilite des explications du modele."
        )

    except ImportError:
        st.warning(
            "Le module `shap` n'est pas installe. "
            "Installez-le avec `pip install shap` pour activer le SHAP dynamique."
        )
    except Exception as e:
        st.error(f"Erreur lors du calcul SHAP dynamique : {e}")
