"""
Score de confiance des predictions.

Combine : qualite des features, marge de prediction, accord des modeles.
"""

import numpy as np


def compute_confidence_scores(probas_rf, probas_xgb=None, if_scores=None,
                              feature_quality=None, n_features_total=27):
    """Calcule un score de confiance [0, 1] pour chaque session.

    Facteurs :
    - prediction_margin : distance a 0.5 (plus loin = plus confiant)
    - feature_quality : proportion de features reellement calculees
    - model_agreement : accord entre RF, XGBoost et IF
    """
    n = len(probas_rf)
    scores = np.zeros(n)

    # 1. Marge de prediction (50% du score)
    margin = np.abs(probas_rf - 0.5) * 2  # 0 a 0.5 → 0 a 1
    scores += 0.50 * margin

    # 2. Qualite des features (25% du score)
    if feature_quality is not None:
        avail = feature_quality.get("available", n_features_total)
        fq_score = avail / n_features_total
    else:
        fq_score = 1.0
    scores += 0.25 * fq_score

    # 3. Accord des modeles (25% du score)
    model_count = 1  # RF toujours present
    agreement_sum = np.zeros(n)

    if probas_xgb is not None:
        model_count += 1
        # Accord RF vs XGBoost : 1 si meme prediction, 0 si oppose
        agreement_sum += 1.0 - np.abs(probas_rf - probas_xgb)

    if if_scores is not None:
        model_count += 1
        # IF : score negatif = anomalie → mapper en binaire
        if_suspect = (if_scores < 0).astype(float)
        rf_suspect = (probas_rf >= 0.5).astype(float)
        agreement_sum += 1.0 - np.abs(if_suspect - rf_suspect)

    if model_count > 1:
        scores += 0.25 * (agreement_sum / (model_count - 1))
    else:
        # Pas d'autres modeles → boost la marge
        scores += 0.25 * margin

    return np.clip(scores, 0, 1)


def confidence_label(score):
    """Retourne (label, couleur) pour un score de confiance."""
    if score >= 0.8:
        return "Tres fiable", "green"
    elif score >= 0.6:
        return "Fiable", "blue"
    elif score >= 0.4:
        return "Moyen", "yellow"
    else:
        return "Faible", "red"
