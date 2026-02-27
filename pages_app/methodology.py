"""
Page : Methodologie — Selection des features (Cohen's d + correlation de Pearson).
Pipeline de reduction : 280 colonnes -> ~60 (Cohen's d) -> 27 (Pearson).
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "methodology_stats.json")


def _load_methodology_data():
    """Charge les statistiques de methodologie depuis le fichier JSON."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render():
    st.header("Methodologie de selection des features")

    data = _load_methodology_data()
    pipeline = data["pipeline"]
    features = data["features"]
    type_dist = data["type_distribution"]
    cat_dist = data["category_distribution"]
    corr_example = data["correlation_example"]

    # =========================================================================
    # 1. INTRODUCTION
    # =========================================================================
    explain(
        "Le dataset CIC-Darknet2020 contient <strong>280 colonnes</strong> par session reseau. "
        "Utiliser toutes ces colonnes directement poserait plusieurs problemes : "
        "surajustement (overfitting), temps de calcul eleve, et difficulte d'interpretation. "
        "Il faut donc <strong>selectionner les features les plus discriminantes</strong> "
        "tout en eliminant la redondance. "
        "Cette page presente le pipeline de selection en deux etapes : "
        "<strong>Cohen's d</strong> (pouvoir discriminant) puis <strong>correlation de Pearson</strong> "
        "(elimination de la redondance)."
    )

    st.markdown("---")

    # =========================================================================
    # 2. PIPELINE VISUEL : 280 -> ~60 -> 27
    # =========================================================================
    st.subheader("Pipeline de reduction dimensionnelle")

    explain(
        "Les 280 colonnes originales se decomposent en <strong>125 features de base</strong>, "
        f"<strong>{pipeline['enc_columns']} features chiffrees (enc)</strong> et "
        f"<strong>{pipeline['ratio_columns']} ratios</strong>, plus "
        f"{pipeline['special_columns']} colonnes speciales (label, identifiant). "
        "Le pipeline en deux etapes reduit ce nombre a <strong>27 features finales</strong>."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Colonnes initiales", str(pipeline["total_columns"]), "blue")
        st.caption("Dataset CIC-Darknet2020 complet (base + enc + ratio)")
    with col2:
        render_metric_card(
            f"Apres Cohen's d (seuil {pipeline['cohens_d_threshold']})",
            f"~{pipeline['cohens_d_candidates']}",
            "yellow"
        )
        st.caption("Features avec un pouvoir discriminant suffisant")
    with col3:
        render_metric_card(
            f"Apres Pearson (seuil {pipeline['pearson_threshold']})",
            str(pipeline["final_features"]),
            "green"
        )
        st.caption("Features finales retenues pour le modele")

    # Fleches visuelles du pipeline
    st.markdown(
        '<div style="text-align:center; color:#64748b; font-size:1.1rem; margin: 8px 0 18px 0;">'
        '280 colonnes &nbsp;&rarr;&nbsp; '
        f'<span style="color:#f59e0b;">Cohen\'s d &ge; {pipeline["cohens_d_threshold"]}</span>'
        ' &nbsp;&rarr;&nbsp; ~60 &nbsp;&rarr;&nbsp; '
        f'<span style="color:#10b981;">Pearson &lt; {pipeline["pearson_threshold"]}</span>'
        ' &nbsp;&rarr;&nbsp; <strong style="color:#3b82f6;">27 features</strong>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================================================================
    # 3. COHEN'S D
    # =========================================================================
    st.subheader("Etape 1 : Cohen's d — Pouvoir discriminant")

    explain(
        "Le <strong>d de Cohen</strong> mesure la taille de l'effet entre deux groupes "
        "(benin vs malveillant). Il se calcule comme la difference des moyennes divisee "
        "par l'ecart-type commun :<br><br>"
        '<div style="text-align:center; font-size:1.05rem; margin:8px 0;">'
        "<em>d = (moyenne_malveillant - moyenne_benin) / ecart_type_commun</em>"
        "</div><br>"
        "Un <strong>d &ge; 0.45</strong> indique une difference notable entre les deux classes. "
        "Seules les features depassant ce seuil sont conservees (~60 sur 280). "
        "Plus le d est eleve, plus la feature separe bien benin et malveillant."
    )

    # Bar chart horizontal des 27 features triees par d
    features_sorted = sorted(features, key=lambda f: f["cohens_d"])

    colors = []
    for f in features_sorted:
        d = f["cohens_d"]
        if d >= 0.8:
            colors.append("#10b981")  # vert — tres discriminant
        elif d >= 0.6:
            colors.append("#3b82f6")  # bleu — bon
        else:
            colors.append("#f59e0b")  # jaune — acceptable

    fig_cohens = go.Figure(go.Bar(
        x=[f["cohens_d"] for f in features_sorted],
        y=[f["name"] for f in features_sorted],
        orientation="h",
        marker_color=colors,
        text=[f'{f["cohens_d"]:.3f}' for f in features_sorted],
        textposition="outside",
        textfont=dict(size=11)
    ))

    fig_cohens.add_vline(
        x=pipeline["cohens_d_threshold"],
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Seuil = {pipeline['cohens_d_threshold']}",
        annotation_position="top right",
        annotation_font_color="#ef4444"
    )

    fig_cohens.update_layout(
        title="Cohen's d des 27 features retenues",
        xaxis_title="Cohen's d (taille de l'effet)",
        yaxis_title="",
        template="plotly_dark",
        height=max(500, len(features_sorted) * 28),
        margin=dict(l=350, t=50, r=80, b=40),
        yaxis=dict(tickfont=dict(size=11))
    )

    st.plotly_chart(fig_cohens, use_container_width=True)

    explain(
        "<strong>Vert</strong> : d &ge; 0.8 (tres discriminant) &mdash; "
        "<strong>Bleu</strong> : 0.6 &le; d &lt; 0.8 (bon) &mdash; "
        "<strong>Jaune</strong> : 0.45 &le; d &lt; 0.6 (acceptable). "
        "La feature la plus discriminante est "
        f"<strong>{features_sorted[-1]['name']}</strong> (d = {features_sorted[-1]['cohens_d']:.3f})."
    )

    st.markdown("---")

    # =========================================================================
    # 4. CORRELATION DE PEARSON
    # =========================================================================
    st.subheader("Etape 2 : Correlation de Pearson — Elimination de la redondance")

    explain(
        "Parmi les ~60 features retenues par Cohen's d, certaines sont <strong>fortement correlees</strong> "
        "entre elles (r de Pearson > 0.85). Garder des features redondantes n'ameliore pas le modele "
        "et peut meme le degrader. "
        "Quand deux features sont trop correlees, on conserve celle avec le <strong>Cohen's d le plus eleve</strong> "
        "et on elimine l'autre."
    )

    # Exemple concret : std_forward_packet_length
    st.markdown("#### Exemple concret : `std_forward_packet_length`")

    explain(
        f"La feature <strong>{corr_example['feature']}</strong> existe en 3 variantes dans le dataset : "
        "base, enc (chiffree) et ratio. Voici comment le filtre de Pearson fonctionne sur cet exemple."
    )

    col_base, col_enc, col_ratio = st.columns(3)

    with col_base:
        status_color = "green" if corr_example["base"]["status"] == "ACCEPTEE" else "red"
        render_metric_card(
            f"Base (col {corr_example['base']['col']})",
            f"d = {corr_example['base']['d']:.2f}",
            status_color
        )
        st.markdown(
            f"<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
            f"Reference<br>"
            f"<strong style='color:{('#10b981' if status_color == 'green' else '#ef4444')};'>"
            f"{corr_example['base']['status']}</strong></div>",
            unsafe_allow_html=True
        )

    with col_enc:
        status_color = "green" if corr_example["enc"]["status"] == "ACCEPTEE" else "red"
        render_metric_card(
            f"Enc (col {corr_example['enc']['col']})",
            f"d = {corr_example['enc']['d']:.2f}",
            status_color
        )
        st.markdown(
            f"<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
            f"r = {corr_example['enc']['r_with_base']:.2f} avec base<br>"
            f"<strong style='color:{('#10b981' if status_color == 'green' else '#ef4444')};'>"
            f"{corr_example['enc']['status']}</strong></div>",
            unsafe_allow_html=True
        )

    with col_ratio:
        status_color = "green" if corr_example["ratio"]["status"] == "ACCEPTEE" else "red"
        render_metric_card(
            f"Ratio (col {corr_example['ratio']['col']})",
            f"d = {corr_example['ratio']['d']:.2f}",
            status_color
        )
        st.markdown(
            f"<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
            f"r = {corr_example['ratio']['r_with_base']:.2f} avec base<br>"
            f"<strong style='color:{('#10b981' if status_color == 'green' else '#ef4444')};'>"
            f"{corr_example['ratio']['status']}</strong></div>",
            unsafe_allow_html=True
        )

    explain(
        f"La variante <strong>enc</strong> (col {corr_example['enc']['col']}) a une correlation de "
        f"<strong>r = {corr_example['enc']['r_with_base']:.2f}</strong> avec la variante base "
        f"(&gt; seuil de {pipeline['pearson_threshold']}). "
        f"Elle est donc <strong>rejetee</strong> car redondante. "
        f"La variante <strong>ratio</strong> (col {corr_example['ratio']['col']}) a une correlation de "
        f"<strong>r = {corr_example['ratio']['r_with_base']:.2f}</strong> "
        f"(&lt; seuil de {pipeline['pearson_threshold']}) : "
        "elle apporte une information complementaire et est <strong>conservee</strong>."
    )

    st.markdown("---")

    # =========================================================================
    # 5. TABLEAU DES 27 FEATURES
    # =========================================================================
    st.subheader("Les 27 features selectionnees")

    explain(
        "Ce tableau recapitule les 27 features retenues apres le pipeline complet. "
        "Chaque feature est identifiee par son <strong>numero de colonne</strong> dans le dataset original, "
        "son <strong>type</strong> (base, enc ou ratio) et sa <strong>categorie fonctionnelle</strong> "
        "(TCP, TTL, Volume, IP, Timing)."
    )

    df_features = pd.DataFrame(features)
    df_features = df_features.rename(columns={
        "col": "Colonne",
        "name": "Nom de la feature",
        "type": "Type",
        "category": "Categorie",
        "cohens_d": "Cohen's d"
    })
    df_features = df_features.sort_values("Cohen's d", ascending=False).reset_index(drop=True)
    df_features.index = df_features.index + 1
    df_features.index.name = "#"

    st.dataframe(
        df_features.style.background_gradient(
            subset=["Cohen's d"], cmap="YlGnBu", vmin=0.4, vmax=1.1
        ).format({"Cohen's d": "{:.3f}"}),
        use_container_width=True,
        height=min(1000, 40 + len(features) * 35)
    )

    st.markdown("---")

    # =========================================================================
    # 6. DISTRIBUTIONS : TYPE ET CATEGORIE
    # =========================================================================
    st.subheader("Repartition des features")

    explain(
        "Les 27 features se repartissent en 3 <strong>types</strong> "
        "(base, enc, ratio — selon la transformation appliquee) "
        "et 5 <strong>categories fonctionnelles</strong> "
        "(TCP, TTL, Volume, IP, Timing — selon l'aspect du trafic mesure)."
    )

    col_pie1, col_pie2 = st.columns(2)

    # --- Pie chart par type ---
    with col_pie1:
        type_labels = list(type_dist.keys())
        type_values = list(type_dist.values())
        type_colors = ["#3b82f6", "#10b981", "#f59e0b"]

        fig_type = go.Figure(go.Pie(
            labels=type_labels,
            values=type_values,
            marker=dict(colors=type_colors),
            textinfo="label+value+percent",
            textfont=dict(size=13),
            hole=0.4,
            hovertemplate="<b>%{label}</b><br>%{value} features<br>%{percent}<extra></extra>"
        ))
        fig_type.update_layout(
            title="Repartition par type",
            template="plotly_dark",
            height=400,
            margin=dict(t=50, b=30, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_type, use_container_width=True)

        explain(
            f"<strong>Ratio</strong> ({type_dist['ratio']}) : rapport entre trafic forward et backward. "
            f"<strong>Base</strong> ({type_dist['base']}) : features originales du dataset. "
            f"<strong>Enc</strong> ({type_dist['enc']}) : features calculees sur le trafic chiffre uniquement."
        )

    # --- Pie chart par categorie ---
    with col_pie2:
        cat_labels = list(cat_dist.keys())
        cat_values = list(cat_dist.values())
        cat_colors = ["#ef4444", "#8b5cf6", "#3b82f6", "#10b981", "#f59e0b"]

        fig_cat = go.Figure(go.Pie(
            labels=cat_labels,
            values=cat_values,
            marker=dict(colors=cat_colors),
            textinfo="label+value+percent",
            textfont=dict(size=13),
            hole=0.4,
            hovertemplate="<b>%{label}</b><br>%{value} features<br>%{percent}<extra></extra>"
        ))
        fig_cat.update_layout(
            title="Repartition par categorie",
            template="plotly_dark",
            height=400,
            margin=dict(t=50, b=30, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        explain(
            f"<strong>TCP</strong> ({cat_dist['TCP']}) : fenetres TCP, headers. "
            f"<strong>Volume</strong> ({cat_dist['Volume']}) : taille des paquets. "
            f"<strong>TTL</strong> ({cat_dist['TTL']}) : time-to-live. "
            f"<strong>IP</strong> ({cat_dist['IP']}) : longueur des paquets IP, ratios. "
            f"<strong>Timing</strong> ({cat_dist['Timing']}) : intervalles d'arrivee, duree du flux."
        )
