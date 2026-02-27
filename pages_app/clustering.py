"""
Page Clustering — Analyse de clustering des sessions malveillantes (K-Means + DBSCAN).
Resultats pre-calcules depuis clustering_stats.json + clustering dynamique si donnees chargees.
"""

import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card, has_labels


# =============================================================================
# Chemins vers les donnees pre-calculees
# =============================================================================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATS_PATH = os.path.join(_BASE_DIR, "data", "clustering_stats.json")
_IMG_DIR = os.path.join(_BASE_DIR, "data", "images")
_IMG_SILHOUETTE = os.path.join(_IMG_DIR, "clustering_silhouette.png")
_IMG_MALWARE = os.path.join(_IMG_DIR, "clustering_malware.png")


def _load_clustering_stats():
    """Charge les statistiques de clustering pre-calculees."""
    if not os.path.exists(_STATS_PATH):
        return None
    with open(_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Fonction principale
# =============================================================================

def render(models, session_features, config):
    st.header("Clustering des sessions malveillantes")

    st.markdown(
        "Cette page presente l'analyse de **clustering** appliquee aux sessions classifiees "
        "comme malveillantes par le Random Forest. L'objectif est de decouvrir des **sous-groupes** "
        "de comportements malveillants et de comprendre ou se concentrent les **faux negatifs** "
        "du modele de classification."
    )

    explain(
        "Le clustering est une methode d'<strong>apprentissage non supervise</strong> qui regroupe "
        "les sessions similaires sans utiliser de labels. En l'appliquant aux sessions malveillantes, "
        "on identifie des <strong>familles de malwares</strong> ayant des comportements reseau distincts. "
        "Cela aide a comprendre pourquoi certaines sessions echappent a la detection (faux negatifs)."
    )

    # --- Charger les statistiques pre-calculees ---
    stats = _load_clustering_stats()
    if stats is None:
        st.warning(
            "Fichier `clustering_stats.json` introuvable. "
            "Les resultats pre-calcules ne peuvent pas etre affiches."
        )
    else:
        _render_static_analysis(stats)

    # --- Clustering dynamique si donnees chargees ---
    st.markdown("---")
    _render_dynamic_clustering(session_features)


# =============================================================================
# Section statique : resultats pre-calcules
# =============================================================================

def _render_static_analysis(stats):
    """Affiche les resultats pre-calcules du clustering."""
    data_info = stats["data"]
    kmeans_info = stats["kmeans"]
    dbscan_info = stats["dbscan"]

    # --- Section 1 : Objectif et methodes ---
    st.markdown("---")
    st.subheader("Objectif et methodes")

    explain(
        f"L'analyse porte sur un echantillon de <strong>{data_info['sample_used']:,}</strong> sessions "
        f"malveillantes (sur {data_info['total_malicious']:,} au total), dont "
        f"<strong>{data_info['fn_count']}</strong> sont des faux negatifs du Random Forest. "
        "Deux algorithmes de clustering sont compares pour identifier les sous-groupes."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**K-Means**\n\n"
            "Algorithme de partitionnement qui divise les donnees en **K clusters** "
            "de taille similaire. Chaque session est affectee au cluster dont le "
            "centroide est le plus proche. Le nombre optimal de clusters K est "
            "determine par le **score de silhouette**."
        )
    with col2:
        st.markdown(
            "**DBSCAN**\n\n"
            "Algorithme base sur la **densite** qui detecte des clusters de formes "
            "arbitraires. Il identifie aussi les **points de bruit** (sessions isolees "
            "qui n'appartiennent a aucun cluster). Deux parametres : `eps` (rayon de "
            "voisinage) et `min_samples` (densite minimale)."
        )

    # --- Section 2 : Selection K-Means ---
    st.markdown("---")
    st.subheader("Selection du nombre de clusters (K-Means)")

    explain(
        "Le <strong>score de silhouette</strong> mesure la qualite du clustering : "
        "un score proche de 1 signifie que les clusters sont bien separes, "
        "un score proche de 0 que les sessions sont a la frontiere entre deux clusters. "
        f"Le meilleur score est obtenu pour <strong>K={kmeans_info['optimal_k']}</strong> "
        f"(silhouette = {kmeans_info['best_silhouette']:.4f})."
    )

    # Image : elbow + silhouette
    if os.path.exists(_IMG_SILHOUETTE):
        st.image(_IMG_SILHOUETTE, use_container_width=True)
    else:
        st.info("Image `clustering_silhouette.png` non disponible.")

    # Tableau des scores de silhouette
    st.markdown("**Scores de silhouette par valeur de K :**")
    sil_scores = kmeans_info["silhouette_scores"]
    sil_df = pd.DataFrame({
        "K (clusters)": [int(k) for k in sil_scores.keys()],
        "Score de silhouette": [float(v) for v in sil_scores.values()]
    })
    sil_df = sil_df.sort_values("K (clusters)")

    # Graphique Plotly des scores de silhouette
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=sil_df["K (clusters)"],
        y=sil_df["Score de silhouette"],
        mode="lines+markers",
        marker=dict(size=10, color="#3b82f6"),
        line=dict(color="#3b82f6", width=2),
        name="Silhouette"
    ))
    # Mettre en evidence le K optimal
    fig_sil.add_trace(go.Scatter(
        x=[kmeans_info["optimal_k"]],
        y=[kmeans_info["best_silhouette"]],
        mode="markers",
        marker=dict(size=16, color="#10b981", symbol="star"),
        name=f"K optimal = {kmeans_info['optimal_k']}"
    ))
    fig_sil.update_layout(
        xaxis_title="Nombre de clusters (K)",
        yaxis_title="Score de silhouette",
        template="plotly_dark",
        height=350,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig_sil, use_container_width=True)

    # Tableau
    st.dataframe(
        sil_df.style.highlight_max(subset=["Score de silhouette"], color="rgba(16, 185, 129, 0.3)")
        .format({"Score de silhouette": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    # --- Section 3 : Resultats K-Means ---
    st.markdown("---")
    st.subheader("Resultats K-Means (K=2)")

    explain(
        "Le K-Means avec K=2 revele deux profils distincts de sessions malveillantes. "
        "Le <strong>Cluster 0</strong> contient la majorite des sessions et concentre "
        "la quasi-totalite des faux negatifs. Le <strong>Cluster 1</strong> correspond "
        "a un malware plus 'agressif' et donc plus facile a detecter."
    )

    # Image : t-SNE K-Means
    if os.path.exists(_IMG_MALWARE):
        st.image(_IMG_MALWARE, use_container_width=True)
    else:
        st.info("Image `clustering_malware.png` non disponible.")

    # Deux colonnes pour les clusters
    clusters = kmeans_info["clusters"]
    col_c0, col_c1 = st.columns(2)

    with col_c0:
        c0 = clusters[0]
        st.markdown(f"**Cluster {c0['id']} — Malware 'standard'**")
        render_metric_card("Sessions", f"{c0['sessions']:,} ({c0['pct']}%)", "blue")
        render_metric_card("Faux negatifs", f"{c0['fn']} ({c0['fn_pct']}%)", "red")
        explain(f"{c0['profile']}")

    with col_c1:
        c1 = clusters[1]
        st.markdown(f"**Cluster {c1['id']} — Malware 'agressif'**")
        render_metric_card("Sessions", f"{c1['sessions']:,} ({c1['pct']}%)", "blue")
        render_metric_card("Faux negatifs", f"{c1['fn']} ({c1['fn_pct']}%)", "green")
        explain(f"{c1['profile']}")

    # --- Section 4 : Resultats DBSCAN ---
    st.markdown("---")
    st.subheader("Resultats DBSCAN")

    explain(
        "DBSCAN identifie des clusters de tailles et formes variables, plus adapte "
        "aux donnees reseau qui ne forment pas toujours des groupes spheriques. "
        f"Avec eps={dbscan_info['eps']} et min_samples={dbscan_info['min_samples']}, "
        f"il trouve <strong>{dbscan_info['clusters']} clusters</strong> et "
        f"<strong>{dbscan_info['noise_points']} points de bruit</strong> ({dbscan_info['noise_pct']}%)."
    )

    # 4 metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Clusters trouves", f"{dbscan_info['clusters']}", "blue")
    with col2:
        render_metric_card("Points de bruit", f"{dbscan_info['noise_points']}", "yellow")
    with col3:
        render_metric_card("Epsilon (eps)", f"{dbscan_info['eps']}", "blue")
    with col4:
        render_metric_card("Min. samples", f"{dbscan_info['min_samples']}", "blue")

    # Tableau comparatif K-Means vs DBSCAN
    st.markdown("**Comparaison K-Means vs DBSCAN :**")

    comparison_df = pd.DataFrame({
        "Critere": [
            "Nombre de clusters",
            "Methode",
            "Gestion du bruit",
            "Forme des clusters",
            "Parametre principal",
            "Score de silhouette"
        ],
        "K-Means": [
            f"{kmeans_info['optimal_k']}",
            "Partitionnement (centroides)",
            "Aucune (toutes les sessions sont affectees)",
            "Spheriques (taille similaire)",
            f"K = {kmeans_info['optimal_k']}",
            f"{kmeans_info['best_silhouette']:.4f}"
        ],
        "DBSCAN": [
            f"{dbscan_info['clusters']}",
            "Densite (voisinage)",
            f"{dbscan_info['noise_points']} points de bruit ({dbscan_info['noise_pct']}%)",
            "Arbitraires (tailles variables)",
            f"eps = {dbscan_info['eps']}, min_samples = {dbscan_info['min_samples']}",
            "Non applicable (densite)"
        ]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # --- Section 5 : Insight cle ---
    st.markdown("---")
    st.subheader("Constat principal")

    fn_cluster0 = clusters[0]
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
                    border: 1px solid rgba(239, 68, 68, 0.4);
                    border-radius: 12px;
                    padding: 20px 24px;
                    margin: 10px 0 20px 0;
                    text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: #ef4444; margin-bottom: 8px;">
                {fn_cluster0['fn_pct']}% des faux negatifs sont dans le Cluster 0
            </div>
            <div style="color: #94a3b8; font-size: 1rem;">
                Le Cluster 0 (malware 'standard') concentre {fn_cluster0['fn']} des {data_info['fn_count']}
                faux negatifs. Ces sessions malveillantes imitent le trafic benin
                (TCP windows elevees, TTL similaire) et sont donc plus difficiles a detecter.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    explain(
        "Ce resultat suggere que les faux negatifs ne sont pas repartis aleatoirement : "
        "ils se concentrent dans un profil specifique de malware qui <strong>imite le trafic normal</strong>. "
        "Pour ameliorer la detection, il faudrait cibler des features supplementaires "
        "capables de distinguer ce sous-groupe du trafic benin."
    )


# =============================================================================
# Section dynamique : clustering en temps reel sur les donnees chargees
# =============================================================================

def _render_dynamic_clustering(session_features):
    """Clustering dynamique sur les donnees chargees (si disponibles et labelisees)."""
    st.subheader("Clustering dynamique")

    if "data" not in st.session_state or "probas" not in st.session_state:
        st.info(
            "Chargez des donnees depuis la page **Vue d'ensemble** pour executer "
            "un clustering dynamique sur vos sessions."
        )
        return

    if not has_labels():
        st.info(
            "Le clustering dynamique necessite des **labels** (verite terrain) pour "
            "isoler les sessions malveillantes. Utilisez les **donnees de demonstration** "
            "pour explorer cette fonctionnalite."
        )
        return

    explain(
        "Cette section applique un clustering <strong>K-Means (K=2)</strong> en temps reel "
        "sur les sessions malveillantes de vos donnees. Les resultats sont projetes en 2D "
        "via <strong>PCA</strong> pour visualiser la separation des clusters."
    )

    df = st.session_state["data"]
    y_true = st.session_state["y_true"]
    preds = st.session_state["preds"]

    # Isoler les sessions malveillantes
    mal_mask = (y_true == 1)
    n_malicious = int(mal_mask.sum())

    if n_malicious < 10:
        st.warning(
            f"Seulement {n_malicious} sessions malveillantes detectees. "
            "Le clustering necessite au minimum 10 sessions pour etre pertinent."
        )
        return

    # Extraire les features des sessions malveillantes
    avail_features = [f for f in session_features if f in df.columns]
    if not avail_features:
        st.error("Aucune feature de session disponible pour le clustering.")
        return

    X_mal = np.nan_to_num(
        df.loc[mal_mask, avail_features].values.astype(float), nan=0.0
    )
    preds_mal = preds[mal_mask]

    # Faux negatifs parmi les malveillantes
    fn_mask_mal = (preds_mal == 0)
    n_fn = int(fn_mask_mal.sum())

    st.markdown(
        f"**{n_malicious:,}** sessions malveillantes identifiees, "
        f"dont **{n_fn}** faux negatifs ({100 * n_fn / max(n_malicious, 1):.1f}%)."
    )

    # --- K-Means ---
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Standardiser pour le clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mal)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Statistiques par cluster
    st.markdown("---")
    st.markdown("**Resultats du clustering K-Means (K=2) :**")

    col_c0, col_c1 = st.columns(2)
    for cluster_id, col in zip([0, 1], [col_c0, col_c1]):
        with col:
            mask_c = (cluster_labels == cluster_id)
            n_sessions_c = int(mask_c.sum())
            pct_c = 100 * n_sessions_c / max(n_malicious, 1)
            n_fn_c = int((fn_mask_mal & mask_c).sum())
            fn_pct_c = 100 * n_fn_c / max(n_fn, 1) if n_fn > 0 else 0.0

            st.markdown(f"**Cluster {cluster_id}**")
            render_metric_card(
                "Sessions",
                f"{n_sessions_c:,} ({pct_c:.1f}%)",
                "blue"
            )
            color_fn = "red" if fn_pct_c > 50 else "green"
            render_metric_card(
                "Faux negatifs",
                f"{n_fn_c} ({fn_pct_c:.1f}%)",
                color_fn
            )

    # --- Projection PCA ---
    st.markdown("---")
    st.markdown("**Projection PCA des clusters :**")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    explain(
        f"Les deux premieres composantes principales expliquent "
        f"<strong>{100 * (explained_var[0] + explained_var[1]):.1f}%</strong> de la variance "
        f"(PC1 = {100 * explained_var[0]:.1f}%, PC2 = {100 * explained_var[1]:.1f}%)."
    )

    # Construire le scatter plot Plotly
    fig = go.Figure()

    colors_cluster = {0: "#3b82f6", 1: "#f59e0b"}
    names_cluster = {0: "Cluster 0", 1: "Cluster 1"}

    for cid in [0, 1]:
        mask_c = (cluster_labels == cid) & (~fn_mask_mal)
        if mask_c.sum() > 0:
            fig.add_trace(go.Scattergl(
                x=X_pca[mask_c, 0],
                y=X_pca[mask_c, 1],
                mode="markers",
                marker=dict(size=4, color=colors_cluster[cid], opacity=0.5),
                name=names_cluster[cid],
                hovertemplate=(
                    f"Cluster {cid}<br>"
                    "PC1: %{x:.2f}<br>"
                    "PC2: %{y:.2f}<extra></extra>"
                )
            ))

    # Faux negatifs en surbrillance
    if n_fn > 0:
        fig.add_trace(go.Scattergl(
            x=X_pca[fn_mask_mal, 0],
            y=X_pca[fn_mask_mal, 1],
            mode="markers",
            marker=dict(
                size=7,
                color="#ef4444",
                symbol="x",
                opacity=0.8,
                line=dict(width=1, color="#ef4444")
            ),
            name=f"Faux negatifs ({n_fn})",
            hovertemplate=(
                "Faux negatif<br>"
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        xaxis_title=f"PC1 ({100 * explained_var[0]:.1f}% variance)",
        yaxis_title=f"PC2 ({100 * explained_var[1]:.1f}% variance)",
        template="plotly_dark",
        height=500,
        margin=dict(t=30, b=30),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    explain(
        "Chaque point represente une session malveillante. Les couleurs indiquent le cluster "
        "d'appartenance (K-Means). Les <strong>croix rouges</strong> marquent les faux negatifs "
        "(sessions malveillantes non detectees par le modele). Observer dans quel cluster "
        "se concentrent les faux negatifs aide a comprendre le profil de malware qui echappe a la detection."
    )
