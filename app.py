"""
Micro-SIEM Dashboard - Classification du trafic reseau chiffre.
Interface Streamlit pour l'analyse du trafic reseau avec les modeles
pre-entraines (Random Forest, XGBoost, Isolation Forest).

Deploiement : Streamlit Cloud
Repository autonome avec modeles et donnees de demonstration integres.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# === PAGE CONFIG (doit etre le premier appel Streamlit) ===
st.set_page_config(
    page_title="Micro-SIEM | Trafic Chiffre",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === PATHS (relatifs a la racine du projet) ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))

RF_SESSION_PATH = os.path.join(APP_DIR, "models", "model_random_forest.joblib")
XGB_PATH = os.path.join(APP_DIR, "models", "model_xgboost.joblib")
IF_PATH = os.path.join(APP_DIR, "models", "model_isolation_forest.joblib")
SESSION_MAPPING_PATH = os.path.join(APP_DIR, "data", "feature_mapping.txt")
PACKET_MAPPING_PATH = os.path.join(APP_DIR, "data", "packet_feature_mapping.txt")
DEMO_DATA_PATH = os.path.join(APP_DIR, "data", "demo_sample.csv")

# === CSS CUSTOM ===
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-card .value.green { color: #10b981; }
    .metric-card .value.red { color: #ef4444; }
    .metric-card .value.blue { color: #3b82f6; }
    .metric-card .value.yellow { color: #f59e0b; }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .sidebar-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 20px;
        text-align: center;
    }
    .hero-banner h1 {
        color: #e2e8f0;
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .hero-banner p {
        color: #94a3b8;
        font-size: 1rem;
    }
    .explain-box {
        background: rgba(59, 130, 246, 0.08);
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


def explain(text):
    """Affiche un bloc d'explication contextuel."""
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


# === CHARGEMENT MODELES ET MAPPINGS ===

@st.cache_resource
def load_models():
    """Charge tous les modeles disponibles."""
    models = {}
    model_info = []

    if os.path.exists(RF_SESSION_PATH):
        models["rf_session"] = joblib.load(RF_SESSION_PATH)
        size = os.path.getsize(RF_SESSION_PATH) / (1024 * 1024)
        model_info.append(("Random Forest (Session)", f"{size:.1f} Mo", "Charge"))
    else:
        model_info.append(("Random Forest (Session)", "-", "Non trouve"))

    if os.path.exists(XGB_PATH):
        models["xgboost"] = joblib.load(XGB_PATH)
        size = os.path.getsize(XGB_PATH) / (1024 * 1024)
        model_info.append(("XGBoost", f"{size:.1f} Mo", "Charge"))
    else:
        model_info.append(("XGBoost", "-", "Non trouve"))

    if os.path.exists(IF_PATH):
        models["isolation_forest"] = joblib.load(IF_PATH)
        size = os.path.getsize(IF_PATH) / (1024 * 1024)
        model_info.append(("Isolation Forest", f"{size:.1f} Mo", "Charge"))
    else:
        model_info.append(("Isolation Forest", "-", "Non trouve"))

    return models, model_info


@st.cache_data
def load_feature_mapping(path):
    """Charge le mapping des features depuis un fichier."""
    names = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    _, name = line.strip().split(",", 1)
                    names.append(name)
    return names


@st.cache_resource
def load_shap_explainer(_model):
    """Charge le SHAP TreeExplainer pour XGBoost."""
    import shap
    return shap.TreeExplainer(_model)


def render_metric_card(title, value, color="blue"):
    """Affiche une carte metrique stylisee."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value {color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# === PAGE 1 : VUE D'ENSEMBLE ===

def page_overview(models, session_features, config):
    st.markdown("""
    <div class="hero-banner">
        <h1>Micro-SIEM &mdash; Classification du trafic chiffre</h1>
        <p>Cet outil analyse des sessions reseau et detecte le trafic malveillant (C2, exfiltration, ransomware)
        a partir des metadonnees de connexion, <strong>sans dechiffrer le contenu</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Upload ou demo ---
    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Importer un fichier CSV de sessions reseau",
            type=["csv"],
            help="CSV avec les 27 features session-based. Colonne 'label' optionnelle (0=benin, 1=malveillant)."
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        load_demo = st.button("Charger la demo (5 000 sessions)",
                              use_container_width=True,
                              type="primary")

    # Charger les donnees
    if uploaded is not None:
        df = pd.read_csv(uploaded, low_memory=False)
        st.session_state["data"] = df
        st.session_state["data_source"] = uploaded.name
    elif load_demo:
        if os.path.exists(DEMO_DATA_PATH):
            df = pd.read_csv(DEMO_DATA_PATH, low_memory=False)
            st.session_state["data"] = df
            st.session_state["data_source"] = "Demonstration : 5 000 sessions du dataset CIC-Darknet2020"
        else:
            st.error("Fichier de demonstration introuvable.")
            return

    if "data" not in st.session_state:
        st.markdown("---")
        st.markdown("### Comment ca marche ?")
        st.markdown("""
        1. **Importez un CSV** de sessions reseau (ou cliquez sur "Charger la demo")
        2. Le modele **Random Forest** (99.5% accuracy) analyse chaque session
        3. Les sessions suspectes sont signalees avec leur **probabilite de malveillance**
        4. Explorez les alertes dans les autres onglets (SHAP, configuration du seuil, statistiques)
        """)

        st.markdown("### Modeles pre-charges")
        _, model_info = load_models()
        cols = st.columns(3)
        descriptions = [
            "Classification principale — analyse les 27 features de chaque session",
            "Utilisee pour expliquer les predictions (SHAP waterfall)",
            "Detection non supervisee — signale les sessions anormales"
        ]
        for i, (name, size, status) in enumerate(model_info):
            with cols[i]:
                icon = "Charge" if status == "Charge" else "Non trouve"
                color = "green" if status == "Charge" else "red"
                render_metric_card(name, icon, color)
                if size != "-":
                    st.caption(descriptions[i])
        return

    df = st.session_state["data"]
    source = st.session_state.get("data_source", "")
    st.caption(f"Source : {source}")

    # --- Predictions ---
    if "rf_session" not in models:
        st.error("Modele Random Forest non disponible.")
        return

    missing = [f for f in session_features if f not in df.columns]
    if missing:
        st.error(f"Features manquantes dans le CSV : {missing[:5]}...")
        return

    X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)
    model_rf = models["rf_session"]

    probas = model_rf.predict_proba(X)[:, 1]
    threshold = config["threshold"]
    preds = (probas >= threshold).astype(int)

    # Scores IF si active
    if_scores = None
    if_preds = None
    if config["use_if"] and "isolation_forest" in models:
        if_raw = models["isolation_forest"].predict(X)
        if_preds = (if_raw == -1).astype(int)
        if_scores = models["isolation_forest"].decision_function(X)
        preds_combined = ((preds == 1) | (if_preds == 1)).astype(int)
    else:
        preds_combined = preds

    # Sauvegarder en session_state
    st.session_state["probas"] = probas
    st.session_state["preds"] = preds_combined
    st.session_state["preds_rf"] = preds
    st.session_state["if_preds"] = if_preds
    st.session_state["if_scores"] = if_scores
    st.session_state["X"] = X

    has_labels = "label" in df.columns
    if has_labels:
        y_true = df["label"].values.astype(int)
        st.session_state["y_true"] = y_true

    # --- Metriques ---
    n_total = len(df)
    n_alerts = preds_combined.sum()
    n_high = (probas >= 0.8).sum()
    n_benign = n_total - n_alerts

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Sessions analysees", f"{n_total:,}", "blue")
    with col2:
        render_metric_card("Sessions benignes", f"{n_benign:,}", "green")
    with col3:
        render_metric_card("Alertes (suspectes)", f"{n_alerts:,}", "red" if n_alerts > 0 else "green")
    with col4:
        render_metric_card("Alertes critiques (P>0.8)", f"{n_high:,}", "yellow")

    explain(
        f"Le modele a analyse <strong>{n_total:,} sessions</strong> reseau. "
        f"Parmi elles, <strong>{n_alerts:,}</strong> ont une probabilite de malveillance "
        f"superieure au seuil de <strong>{threshold}</strong> (configurable dans l'onglet Configuration). "
        f"Les alertes critiques sont celles avec une probabilite > 0.8."
    )

    # --- Distribution des probabilites ---
    st.subheader("Distribution des probabilites")
    import plotly.graph_objects as go

    if has_labels:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probas[y_true == 0], nbinsx=50, name="Sessions benignes (label=0)",
            marker_color="#3b82f6", opacity=0.6
        ))
        fig.add_trace(go.Histogram(
            x=probas[y_true == 1], nbinsx=50, name="Sessions malveillantes (label=1)",
            marker_color="#ef4444", opacity=0.6
        ))
        fig.update_layout(barmode="overlay")
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probas, nbinsx=50, name="Toutes les sessions",
            marker_color="#3b82f6", opacity=0.7
        ))

    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil = {threshold}")
    fig.update_layout(
        xaxis_title="Probabilite de malveillance (0 = benin, 1 = malveillant)",
        yaxis_title="Nombre de sessions",
        template="plotly_dark", height=350, margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    explain(
        "Chaque session recoit un <strong>score entre 0 et 1</strong>. "
        "Un score proche de 0 = probablement benin. Proche de 1 = probablement malveillant. "
        "La ligne rouge verticale est le seuil de decision : tout ce qui est a droite est signale comme alerte. "
        "Un bon modele separe bien les deux groupes (pics distincts a gauche et a droite)."
    )

    # --- Table des sessions a risque ---
    st.subheader("Top 100 des sessions les plus suspectes")

    df_display = df.copy()
    df_display["probabilite"] = probas
    df_display["verdict"] = np.where(preds_combined == 1, "SUSPECT", "Benin")
    if if_preds is not None:
        df_display["anomalie_IF"] = np.where(if_preds == 1, "OUI", "-")
    if has_labels:
        df_display["verite_terrain"] = np.where(y_true == 1, "Malveillant", "Benin")

    display_cols = ["probabilite", "verdict"]
    if if_preds is not None:
        display_cols.append("anomalie_IF")
    if has_labels:
        display_cols.append("verite_terrain")
    top_features = session_features[:5]
    display_cols.extend([f for f in top_features if f in df_display.columns])

    df_sorted = df_display[display_cols].sort_values("probabilite", ascending=False)

    st.dataframe(
        df_sorted.head(100).style.background_gradient(
            subset=["probabilite"], cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True,
        height=400
    )

    explain(
        "Ce tableau montre les 100 sessions avec la probabilite de malveillance la plus elevee. "
        "La colonne <strong>probabilite</strong> est coloree du vert (benin) au rouge (malveillant). "
        "Si les donnees contiennent des labels, la colonne <strong>verite terrain</strong> montre "
        "la vraie nature de la session pour verifier les predictions."
    )


# === PAGE 2 : ANALYSE DETAILLEE ===

def page_detail(models, session_features, config):
    st.header("Analyse detaillee d'une session")
    st.markdown("Selectionnez une session pour comprendre **pourquoi** le modele la considere comme suspecte ou benigne.")

    if "data" not in st.session_state or "probas" not in st.session_state:
        st.warning("Chargez d'abord des donnees dans **Vue d'ensemble** (cliquez sur 'Charger la demo').")
        return

    df = st.session_state["data"]
    probas = st.session_state["probas"]
    preds = st.session_state["preds"]
    X = st.session_state["X"]

    # --- Selection de session ---
    st.markdown("---")
    n = len(df)
    col_select, col_filter = st.columns([2, 1])
    with col_filter:
        filter_type = st.radio("Filtrer par", ["Toutes", "Alertes uniquement", "Top 50 suspectes"],
                               horizontal=True)
    with col_select:
        if filter_type == "Alertes uniquement":
            alert_indices = np.where(preds == 1)[0]
            if len(alert_indices) == 0:
                st.info("Aucune alerte avec le seuil actuel. Baissez le seuil dans Configuration.")
                return
            idx = st.selectbox("Choisir une session", alert_indices,
                               format_func=lambda i: f"Session {i} — probabilite {probas[i]:.4f}")
        elif filter_type == "Top 50 suspectes":
            top_indices = np.argsort(probas)[::-1][:50]
            idx = st.selectbox("Choisir une session", top_indices,
                               format_func=lambda i: f"Session {i} — probabilite {probas[i]:.4f}")
        else:
            idx = st.number_input("Index de session", min_value=0, max_value=n-1, value=0)

    # --- Verdict ---
    proba = probas[idx]
    verdict = "SUSPECT" if preds[idx] == 1 else "Benin"

    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        color = "red" if proba >= 0.5 else ("yellow" if proba >= 0.3 else "green")
        render_metric_card("Probabilite de malveillance", f"{proba:.4f}", color)
    with col_v2:
        render_metric_card("Verdict du modele", verdict, "red" if verdict == "SUSPECT" else "green")
    with col_v3:
        if "y_true" in st.session_state:
            label = "Malveillant" if st.session_state["y_true"][idx] == 1 else "Benin"
            correct = (st.session_state["y_true"][idx] == preds[idx])
            render_metric_card("Verite terrain", label, "green" if correct else "red")
        else:
            render_metric_card("Verite terrain", "Non disponible", "blue")

    st.markdown("---")

    # --- SHAP explanation ---
    if "xgboost" in models:
        st.subheader("Pourquoi cette prediction ? (Explication SHAP)")

        explain(
            "<strong>SHAP</strong> decompose la prediction en contributions de chaque feature. "
            "Chaque barre montre l'influence d'une caracteristique sur la decision : "
            "<span style='color:#ef4444'>en rouge</span> = pousse vers 'malveillant', "
            "<span style='color:#3b82f6'>en bleu</span> = pousse vers 'benin'. "
            "Les features en haut sont celles qui ont le plus d'impact sur cette session."
        )

        try:
            import shap
            import matplotlib.pyplot as plt

            explainer = load_shap_explainer(models["xgboost"])
            X_session = X[idx:idx+1]
            shap_values = explainer.shap_values(X_session)

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_session[0],
                feature_names=session_features
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Erreur SHAP : {e}")
    else:
        st.info("Modele XGBoost non disponible pour les explications SHAP.")

    st.markdown("---")

    # --- Comparaison feature values ---
    st.subheader("Valeurs des features de cette session")

    explain(
        "Ce tableau compare les valeurs de la session selectionnee avec la moyenne du dataset. "
        "Le <strong>Z-score</strong> mesure l'ecart par rapport a la normale : "
        "un Z-score > 2 (rouge) signifie que la valeur est anormalement elevee, "
        "< -2 (bleu) = anormalement basse. "
        "C'est un indicateur de ce qui rend cette session differente des autres."
    )

    feature_vals = X[idx]
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)

    z_scores = np.where(std_all > 0, (feature_vals - mean_all) / std_all, 0)

    df_features = pd.DataFrame({
        "Feature": session_features,
        "Valeur (cette session)": feature_vals,
        "Moyenne (dataset)": mean_all,
        "Ecart-type": std_all,
        "Z-score": z_scores
    }).sort_values("Z-score", key=abs, ascending=False)

    st.dataframe(
        df_features.style.background_gradient(
            subset=["Z-score"], cmap="RdBu_r", vmin=-3, vmax=3
        ).format({
            "Valeur (cette session)": "{:.4f}",
            "Moyenne (dataset)": "{:.4f}",
            "Ecart-type": "{:.4f}",
            "Z-score": "{:.2f}"
        }),
        use_container_width=True,
        height=500
    )


# === PAGE 3 : CONFIGURATION ===

def page_config(config):
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
        n_alerts = (probas >= threshold).sum()
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


# === PAGE 4 : STATISTIQUES ===

def page_stats(models, session_features, config):
    st.header("Statistiques du modele")
    st.markdown("Performance globale du modele sur les donnees chargees.")

    if "probas" not in st.session_state:
        st.warning("Chargez d'abord des donnees dans **Vue d'ensemble** (cliquez sur 'Charger la demo').")
        return

    import plotly.graph_objects as go

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


# === PAGE A PROPOS ===

def page_about():
    st.header("A propos de cet outil")

    st.markdown("""
    ### Qu'est-ce que c'est ?

    Ce dashboard est un **Micro-SIEM** (Security Information and Event Management) specialise
    dans la detection de trafic reseau malveillant **chiffre**. Il analyse les metadonnees des
    sessions reseau (duree, volume, timing, nombre de paquets) sans avoir besoin de dechiffrer
    le contenu.

    ### Pourquoi c'est utile ?

    Avec plus de 90% du trafic web chiffre en TLS/SSL, les outils classiques d'inspection
    de contenu (DPI) sont aveugles. Ce dashboard utilise le **Machine Learning** pour
    detecter des comportements suspects (C2, exfiltration, ransomware) a partir des
    caracteristiques statistiques des connexions.

    ---

    ### Modeles utilises

    | Modele | Role | Performance |
    |--------|------|-------------|
    | **Random Forest** | Classification principale (27 features) | **F1 = 0.9950** (99.5% accuracy) |
    | **XGBoost** | Explications SHAP par session | F1 = 0.9898 |
    | **Isolation Forest** | Detection d'anomalies non supervisee | Precision = 94.7% |

    ### Donnees d'entrainement

    - **Dataset** : CIC-Darknet2020 (244 000 sessions d'entrainement, 122 000 de test)
    - **Features** : 27 caracteristiques de session selectionnees par importance (Gini + Cohen's d)
    - **Labels** : 0 = trafic benin, 1 = trafic malveillant (Tor, VPN malveillant)

    ---

    ### Pipeline d'analyse complet

    Ce dashboard est l'aboutissement d'un projet en 5 axes :
    1. **Analyse packet-based** — Classification par paquets individuels (99.98% accuracy)
    2. **Comparaison d'algorithmes** — Random Forest vs XGBoost vs MLP
    3. **Interpretabilite** — SHAP : comprendre pourquoi le modele decide
    4. **Visualisation** — t-SNE, UMAP, clustering K-Means des familles de malwares
    5. **Detection d'anomalies** — Isolation Forest non supervise

    ### Technologies

    Streamlit, scikit-learn, XGBoost, SHAP, Plotly, pandas, NumPy, Matplotlib
    """)

    st.markdown("---")
    st.caption("Projet realise par Loris Dietrich — Analyse de cybersecurite")


# === MAIN ===

def main():
    models, model_info = load_models()
    session_features = load_feature_mapping(SESSION_MAPPING_PATH)
    packet_features = load_feature_mapping(PACKET_MAPPING_PATH)

    # --- Sidebar ---
    st.sidebar.markdown('<div class="sidebar-title">Micro-SIEM</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-subtitle">Classification du trafic chiffre</div>',
                        unsafe_allow_html=True)

    page = st.sidebar.radio("Navigation", [
        "Vue d'ensemble",
        "Analyse detaillee",
        "Configuration",
        "Statistiques",
        "A propos"
    ], label_visibility="collapsed")

    st.sidebar.divider()

    # Config rapide dans la sidebar
    st.sidebar.subheader("Parametres rapides")
    threshold = st.sidebar.slider("Seuil de detection", 0.0, 1.0, 0.5, 0.01,
                                  key="sidebar_threshold",
                                  help="Sessions avec P(malveillant) >= seuil = alerte")
    use_if = st.sidebar.toggle("Isolation Forest", False, key="sidebar_if",
                               help="Activer la detection d'anomalies non supervisee")

    config = {"threshold": threshold, "use_if": use_if}

    st.sidebar.divider()

    # Info modeles
    st.sidebar.subheader("Modeles charges")
    for name, size, status in model_info:
        icon = "\u2705" if status == "Charge" else "\u274c"
        st.sidebar.caption(f"{icon} {name} ({size})")

    st.sidebar.divider()
    st.sidebar.caption("Micro-SIEM | Loris Dietrich")

    # --- Pages ---
    if page == "Vue d'ensemble":
        page_overview(models, session_features, config)
    elif page == "Analyse detaillee":
        page_detail(models, session_features, config)
    elif page == "Configuration":
        new_config = page_config(config)
        if new_config:
            config = new_config
    elif page == "Statistiques":
        page_stats(models, session_features, config)
    elif page == "A propos":
        page_about()


if __name__ == "__main__":
    main()
