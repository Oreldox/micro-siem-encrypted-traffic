"""
Detection de derive (drift) des features entre les donnees importees
et les statistiques d'entrainement.

Compare les distributions (mean, std) feature par feature
et produit un score de compatibilite global.
"""

import numpy as np


def compute_feature_drift(X, session_features, training_stats):
    """Compare les features du dataset importe aux stats d'entrainement.

    Retourne un dict avec :
    - drift_scores : array de drift par feature (0 = identique, >1 = forte derive)
    - global_score : score global de compatibilite (0-100, 100 = parfait)
    - drifted_features : liste des features en derive (score > 1.5)
    - category_scores : score moyen par categorie de features
    """
    if training_stats is None or X.shape[0] < 5:
        return None

    train_mean = training_stats["mean"]
    train_std = training_stats["std"]

    # Statistiques du dataset importe
    import_mean = X.mean(axis=0)
    import_std = X.std(axis=0)

    # Drift par feature : distance normalisee des moyennes + ratio des ecarts-types
    safe_std = np.where(train_std > 1e-10, train_std, 1.0)
    mean_drift = np.abs(import_mean - train_mean) / safe_std
    std_ratio = np.where(
        train_std > 1e-10,
        np.maximum(import_std / train_std, train_std / np.maximum(import_std, 1e-10)),
        1.0
    )
    # Score combine : ecart de moyenne + ecart de dispersion
    drift_scores = 0.7 * mean_drift + 0.3 * (std_ratio - 1.0)
    drift_scores = np.clip(drift_scores, 0, 10)

    # Features en derive
    drifted = []
    for i, feat in enumerate(session_features):
        if drift_scores[i] > 1.5:
            drifted.append({
                "name": feat,
                "index": i,
                "drift_score": float(drift_scores[i]),
                "train_mean": float(train_mean[i]),
                "import_mean": float(import_mean[i]),
                "train_std": float(safe_std[i]),
                "import_std": float(import_std[i]),
            })
    drifted.sort(key=lambda x: x["drift_score"], reverse=True)

    # Score global (0-100) : inverse du drift moyen
    avg_drift = float(drift_scores.mean())
    global_score = max(0, min(100, 100 - avg_drift * 20))

    # Scores par categorie
    categories = {
        "Timing": ["max_Interval_of_arrival_time_of_backward_traffic_enc",
                    "max_Interval_of_arrival_time_of_backward_traffic_ratio",
                    "flow_duration_of_backward_traffic_ratio"],
        "Volume": ["min_forward_packet_length", "min_backward_packet_length",
                    "std_forward_packet_length_enc", "std_forward_packet_length_ratio",
                    "mean_forward_packet_length_ratio", "median_forward_packet_length_ratio",
                    "mean_backward_packet_length_ratio"],
        "IP / Ratio": ["IPratio_enc", "IPratio_ratio", "max_length_of_IP_packet_ratio",
                        "std_length_of_IP_packet_ratio", "meidan_length_of_IP_packet_ratio",
                        "median_length_of_IP_packet_ratio", "std_length_of_IP_packet_enc"],
        "TCP": ["max_TCP_windows_size_value_forward_traffic_ratio",
                "total_TCP_windows_size_value_forward_traffic_ratio",
                "std_TCP_windows_size_value_backward_traffic_ratio",
                "median_TCP_windows_size_value_backward_traffic_ratio",
                "max_Change_values_of_TCP_windows_length_per_session",
                "std_Length_of_TCP_packet_header"],
        "TTL": ["min_ttl_forward_traffic", "max_ttl_backward_traffic_enc",
                "median_ttl_backward_traffic", "mean_ttl_backward_traffic_ratio",
                "min_ttl_backward_traffic_ratio"],
    }

    category_scores = {}
    for cat, feats in categories.items():
        indices = [session_features.index(f) for f in feats if f in session_features]
        if indices:
            cat_drift = float(np.mean(drift_scores[indices]))
            category_scores[cat] = max(0, min(100, 100 - cat_drift * 20))

    return {
        "drift_scores": drift_scores,
        "global_score": global_score,
        "drifted_features": drifted,
        "category_scores": category_scores,
        "avg_drift": avg_drift,
    }


def transferability_label(score):
    """Retourne (label, couleur) pour un score de transferabilite."""
    if score >= 80:
        return "Tres compatible", "green"
    elif score >= 60:
        return "Compatible", "blue"
    elif score >= 40:
        return "Partiellement compatible", "yellow"
    else:
        return "Faible compatibilite", "red"
