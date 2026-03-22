"""
Page Accueil — Presentation du projet, modele, dataset, CTA directs.
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go

from src.ui_components import inject_css, explain, render_metric_card, load_demo_data

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_REF = {
    "accuracy": 99.50,
    "precision": 99.84,
    "recall": 99.16,
    "f1": 0.9950,
    "auc": 0.9999,
}


def render():
    inject_css()

    # === Hero banner ===
    st.markdown("""
    <div class="hero-banner">
        <h1>Analyse de Trafic Chiffre par Machine Learning</h1>
        <p>
            Classification automatique de sessions reseau chiffrees
            (benignes vs malveillantes) a l'aide d'un modele <strong>Random Forest</strong>
            entraine sur 488 524 sessions — sans dechiffrer le contenu.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === CTA — Acces rapide ===
    st.subheader("Commencer")

    col_cta1, col_cta2, col_cta3 = st.columns(3)

    with col_cta1:
        st.markdown("""
        <div class="cta-card">
            <h3>Essayer avec un dataset de demo</h3>
            <p>5 000 sessions CIC-Darknet2020 pre-chargees, resultats instantanes</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Lancer la demo", use_container_width=True, type="primary"):
            load_demo_data()
            st.session_state["_loaded_dataset"] = "demo_sample.csv"
            st.success("Donnees de demo chargees. Ouvrez **Tester le modele** dans le menu.")

    with col_cta2:
        st.markdown("""
        <div class="cta-card">
            <h3>Tester sur un autre dataset</h3>
            <p>USTC-TFC2016, dataset custom, ou importez vos propres donnees</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Ouvrez **Tester le modele** dans le menu a gauche.")

    with col_cta3:
        st.markdown("""
        <div class="cta-card">
            <h3>Comprendre la methodologie</h3>
            <p>Pipeline de selection des features : 280 colonnes vers 27 features</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Ouvrez **Methodologie** dans le menu a gauche.")

    st.markdown("---")

    # === Le modele ===
    st.header("Le modele")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _card("Algorithme", "Random Forest", "blue")
    with col2:
        _card("Features", "27 / 280", "blue")
    with col3:
        _card("F1-score", f"{TRAINING_REF['f1']:.4f}", "green")
    with col4:
        _card("Accuracy", f"{TRAINING_REF['accuracy']:.1f}%", "green")
    with col5:
        _card("AUC", f"{TRAINING_REF['auc']:.4f}", "green")

    explain(
        "Le modele est un <strong>Random Forest</strong> entraine sur 488 524 sessions "
        "du dataset CIC-Darknet2020. Il utilise <strong>27 features</strong> selectionnees "
        "parmi 280 colonnes via un pipeline Cohen's d + Pearson. "
        "Il est valide par un <strong>XGBoost</strong> et un <strong>Isolation Forest</strong> "
        "en complement."
    )

    # === Comment ca marche ===
    st.header("Comment ca marche ?")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown("""
        <div class="metric-card">
            <h3>1. Import</h3>
            <div style="color:#3b82f6; font-size:1.5rem; margin:8px 0;">PCAP / CSV</div>
            <p style="color:#94a3b8; font-size:0.8rem;">
                Importez un fichier PCAP, CSV, Excel ou selectionnez un dataset inclus
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown("""
        <div class="metric-card">
            <h3>2. Extraction</h3>
            <div style="color:#f59e0b; font-size:1.5rem; margin:8px 0;">27 features</div>
            <p style="color:#94a3b8; font-size:0.8rem;">
                Les metadonnees reseau sont extraites : timing, volume, TCP, TTL, ratios IP
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown("""
        <div class="metric-card">
            <h3>3. Classification</h3>
            <div style="color:#10b981; font-size:1.5rem; margin:8px 0;">ML</div>
            <p style="color:#94a3b8; font-size:0.8rem;">
                Le Random Forest predit si chaque session est benigne ou malveillante
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_p4:
        st.markdown("""
        <div class="metric-card">
            <h3>4. Analyse</h3>
            <div style="color:#ef4444; font-size:1.5rem; margin:8px 0;">Resultats</div>
            <p style="color:#94a3b8; font-size:0.8rem;">
                Visualisez les alertes, explorez les sessions suspectes, exportez un rapport
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # === Dataset d'entrainement ===
    st.header("Dataset d'entrainement — CIC-Darknet2020")

    stats = _load_methodology_stats()
    if stats is None:
        st.warning("Fichier methodology_stats.json introuvable.")
        return

    ds = stats["dataset"]

    col_info, col_chart = st.columns([3, 2])

    with col_info:
        st.markdown(f"""
        | | Nombre de sessions |
        |---|---:|
        | **Total** | **{ds['total_sessions']:,}** |
        | Entrainement | {ds['train_sessions']:,} |
        | Test | {ds['test_sessions']:,} |
        | Familles de malware | {ds['malware_families']} |
        """)

        st.markdown("**Sources du trafic benin :**")
        rows = ""
        for d in ds["benign_datasets"]:
            rows += f"| {d['name']} | {d['source']} | {d['sessions']:,} |\n"
        st.markdown(f"""
        | Dataset | Source | Sessions |
        |---|---|---:|
        {rows}""")

        st.markdown(f"""
        **Trafic malveillant :** {ds['malicious_sessions']:,} sessions issues
        de **{ds['malware_families']} familles de malware** (trojans, ransomware,
        botnets, spyware, adware).
        """)

    with col_chart:
        fig = go.Figure(data=[go.Pie(
            labels=["Benin", "Malveillant"],
            values=[ds["benign_sessions"], ds["malicious_sessions"]],
            marker=dict(colors=["#3b82f6", "#ef4444"]),
            textinfo="label+percent",
            hole=0.4,
        )])
        fig.update_layout(
            template="plotly_dark", height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Selection des features ===
    st.header("Selection des features")

    pipeline = stats["pipeline"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _card("Colonnes initiales", str(pipeline["total_columns"]), "blue")
    with col_b:
        _card(f"Cohen's d > {pipeline['cohens_d_threshold']}", str(pipeline["cohens_d_candidates"]), "yellow")
    with col_c:
        _card(f"Pearson < {pipeline['pearson_threshold']}", str(pipeline["final_features"]), "green")

    explain(
        f"<strong>280 colonnes</strong> &rarr; filtre Cohen's d (seuil {pipeline['cohens_d_threshold']}) "
        f"&rarr; <strong>{pipeline['cohens_d_candidates']} candidates</strong> &rarr; "
        f"filtre Pearson (seuil {pipeline['pearson_threshold']}) &rarr; "
        f"<strong>{pipeline['final_features']} features finales</strong>. "
        "Le Cohen's d mesure la taille d'effet entre benin et malveillant. "
        "Le filtre Pearson elimine les features redondantes."
    )

    # Top features bar chart
    features = stats["features"][:10]
    fig = go.Figure(go.Bar(
        y=[f["name"].replace("_", " ")[:40] for f in reversed(features)],
        x=[f["cohens_d"] for f in reversed(features)],
        orientation="h",
        marker_color=["#3b82f6"] * len(features),
        text=[f"{f['cohens_d']:.3f}" for f in reversed(features)],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 10 features (Cohen's d)",
        template="plotly_dark", height=400,
        margin=dict(t=40, b=20, l=20, r=60),
        xaxis_title="Cohen's d",
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Limites connues ===
    st.markdown("---")
    st.header("Limites de generalisation")
    explain(
        "Ce modele est specialise sur le trafic du dataset <strong>CIC-Darknet2020</strong>. "
        "Il ne generalise pas automatiquement a d'autres types de malwares ou d'autres methodes "
        "d'extraction de features. Voici les facteurs qui influencent les performances sur un "
        "nouveau dataset :"
    )

    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("""
        **Distribution des features**

        Les 27 features doivent avoir des distributions
        statistiques similaires a celles vues a l'entrainement.
        Un dataset extrait differemment peut avoir des valeurs
        dans des plages differentes.
        """)
    with col_l2:
        st.markdown("""
        **Familles de malware**

        Le modele connait 25 familles de malware (trojans,
        ransomware, botnets). Il peut ne pas detecter des
        familles inconnues ou des attaques zero-day.
        """)
    with col_l3:
        st.markdown("""
        **Epoque des donnees**

        Les patterns reseau evoluent. Un malware de 2024
        peut se comporter differemment d'un malware de 2020
        vu a l'entrainement.
        """)

    explain(
        "La page <strong>Tester le modele</strong> inclut un diagnostic automatique de compatibilite "
        "qui compare les distributions de vos donnees a celles de l'entrainement."
    )


def _card(title, value, color="blue"):
    """Mini metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value {color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def _load_methodology_stats():
    """Charge methodology_stats.json."""
    path = os.path.join(APP_DIR, "data", "methodology_stats.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
