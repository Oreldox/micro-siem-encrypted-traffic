"""
Chargement et gestion des modeles ML.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RF_SESSION_PATH = os.path.join(APP_DIR, "models", "model_random_forest.joblib")
XGB_PATH = os.path.join(APP_DIR, "models", "model_xgboost.joblib")
IF_PATH = os.path.join(APP_DIR, "models", "model_isolation_forest.joblib")
RF_PACKET_PATH = os.path.join(APP_DIR, "models", "model_rf_paquets.joblib")
SESSION_MAPPING_PATH = os.path.join(APP_DIR, "data", "feature_mapping.txt")
PACKET_MAPPING_PATH = os.path.join(APP_DIR, "data", "packet_feature_mapping.txt")
DEMO_DATA_PATH = os.path.join(APP_DIR, "data", "demo_sample.csv")
DEMO_PACKETS_PATH = os.path.join(APP_DIR, "data", "demo_packets_sample.csv")
UMAP_EMBEDDING_PATH = os.path.join(APP_DIR, "data", "demo_umap_embedding.npz")


@st.cache_resource
def load_models():
    """Charge les modeles principaux (RF session, XGBoost, IF)."""
    models = {}
    model_info = []

    for key, path, name in [
        ("rf_session", RF_SESSION_PATH, "Random Forest (Session)"),
        ("xgboost", XGB_PATH, "XGBoost"),
        ("isolation_forest", IF_PATH, "Isolation Forest"),
    ]:
        if os.path.exists(path):
            models[key] = joblib.load(path)
            size = os.path.getsize(path) / (1024 * 1024)
            model_info.append((name, f"{size:.1f} Mo", "Charge"))
        else:
            model_info.append((name, "-", "Non trouve"))

    return models, model_info


@st.cache_resource
def load_rf_packet():
    """Charge le modele RF paquets (lazy — uniquement quand la page cascade est visitee)."""
    if os.path.exists(RF_PACKET_PATH):
        return joblib.load(RF_PACKET_PATH)
    return None


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
    """Charge le SHAP TreeExplainer (compatible RF et XGBoost)."""
    import shap
    return shap.TreeExplainer(_model)


@st.cache_data
def load_training_stats(session_features):
    """Charge ou calcule les statistiques du dataset d'entrainement.

    Retourne {"mean": array, "std": array, "min": array, "max": array}
    ou None si indisponible.
    """
    stats_path = os.path.join(APP_DIR, "data", "training_stats.npz")
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        mean = data["mean"]
        std = data["std"]
        return {
            "mean": mean,
            "std": std,
            "min": data["min"] if "min" in data else mean - 3 * std,
            "max": data["max"] if "max" in data else mean + 3 * std,
        }

    # Calculer depuis les donnees demo
    if os.path.exists(DEMO_DATA_PATH):
        df = pd.read_csv(DEMO_DATA_PATH, low_memory=False)
        feat_list = list(session_features)
        avail = [f for f in feat_list if f in df.columns]
        if len(avail) == len(feat_list):
            X = np.nan_to_num(df[feat_list].values.astype(float), nan=0.0)
            stats = {
                "mean": X.mean(axis=0),
                "std": X.std(axis=0),
                "min": X.min(axis=0),
                "max": X.max(axis=0),
            }
            # Sauvegarder pour eviter le recalcul
            try:
                np.savez(stats_path, **stats)
            except OSError:
                pass
            return stats

    return None
