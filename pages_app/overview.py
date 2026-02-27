"""
Page Accueil — Presentation du projet, modele et dataset d'entrainement.
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go

from src.ui_components import inject_css, explain

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
            (benignes vs malveillantes) a l'aide d'un modele <strong>Random Forest</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

    # === Pipeline de features ===
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
        f"<strong>280 colonnes</strong> → filtre Cohen's d (seuil {pipeline['cohens_d_threshold']}) "
        f"→ <strong>{pipeline['cohens_d_candidates']} candidates</strong> → "
        f"filtre Pearson (seuil {pipeline['pearson_threshold']}) → "
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

    # === CTA ===
    st.markdown("---")
    st.markdown(
        "### Tester le modele sur un dataset\n"
        "Rendez-vous sur la page **Tester le modele** dans le menu a gauche "
        "pour evaluer les performances du modele sur differents datasets "
        "de trafic chiffre."
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
