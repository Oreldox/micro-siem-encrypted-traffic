"""
Module partage de traitement des donnees.
Fournit les utilitaires communs entre overview.py et external_test.py.
"""

import os
import pandas as pd
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_external_test_data():
    """Charge le dataset de test externe inclus (1000 sessions).

    Returns:
        pd.DataFrame ou None si le fichier n'existe pas.
    """
    path = os.path.join(APP_DIR, "data", "external_test_sample.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False)


def prepare_features(df, session_features):
    """Prepare la matrice de features X a partir d'un DataFrame.

    - Verifie les colonnes presentes
    - Ajoute les colonnes manquantes a zero
    - Remplace les NaN par 0

    Returns:
        (X, missing_features) : matrice numpy + liste des features manquantes
    """
    missing = [f for f in session_features if f not in df.columns]
    for f in missing:
        df[f] = 0.0

    X = np.nan_to_num(df[session_features].values.astype(float), nan=0.0)
    return X, missing


def compute_feature_quality(df, session_features):
    """Calcule le nombre de features non-zero dans un DataFrame.

    Returns:
        dict avec 'total', 'available' et 'missing_features'
    """
    available = sum(
        1 for f in session_features
        if f in df.columns and (df[f] != 0).any()
    )
    missing = [f for f in session_features if f not in df.columns or not (df[f] != 0).any()]
    return {
        "total": len(session_features),
        "available": available,
        "missing_features": missing,
    }
