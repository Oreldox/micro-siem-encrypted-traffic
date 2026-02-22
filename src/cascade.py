"""
Mode cascade : analyse session puis paquets pour les sessions incertaines.
"""

import numpy as np
import pandas as pd


def identify_uncertain_sessions(probas, low=0.3, high=0.7):
    """Retourne les indices des sessions dans la zone d'incertitude."""
    mask = (probas >= low) & (probas <= high)
    return np.where(mask)[0]


def predict_packets(model_rf_paquets, packet_df, packet_features):
    """Predit au niveau paquet avec le RF paquets."""
    missing = [f for f in packet_features if f not in packet_df.columns]
    if missing:
        raise ValueError(f"Features manquantes dans les donnees paquets : {missing[:5]}")

    X = np.nan_to_num(packet_df[packet_features].values.astype(float), nan=0.0)
    probas = model_rf_paquets.predict_proba(X)[:, 1]
    preds = model_rf_paquets.predict(X)
    return preds, probas


def aggregate_to_session(packet_preds, packet_probas, session_ids, strategy="conservative"):
    """Agrege les predictions par paquet en verdict par session.

    Strategies :
    - 'vote' : majorite des paquets
    - 'mean_proba' : moyenne des probabilites >= 0.5
    - 'conservative' : si un seul paquet malveillant -> session malveillante
    """
    df = pd.DataFrame({
        "session_id": session_ids,
        "pred": packet_preds,
        "proba": packet_probas
    })

    results = {}
    for sid, group in df.groupby("session_id"):
        if strategy == "vote":
            verdict = 1 if group["pred"].sum() > len(group) / 2 else 0
            confidence = group["pred"].mean()
        elif strategy == "mean_proba":
            mean_p = group["proba"].mean()
            verdict = 1 if mean_p >= 0.5 else 0
            confidence = mean_p
        elif strategy == "conservative":
            verdict = 1 if group["pred"].max() == 1 else 0
            confidence = group["proba"].max()
        else:
            raise ValueError(f"Strategie inconnue : {strategy}")

        results[sid] = {
            "verdict": verdict,
            "confidence": confidence,
            "n_packets": len(group),
            "n_malicious_packets": int(group["pred"].sum()),
            "mean_proba": group["proba"].mean(),
            "max_proba": group["proba"].max(),
        }

    return results


def cascade_analysis(session_probas, session_marks, packet_df,
                     model_rf_paquets, packet_features,
                     low=0.3, high=0.7, strategy="conservative"):
    """Pipeline cascade complet.

    1. Identifier les sessions incertaines (proba entre low et high)
    2. Filtrer les paquets correspondants
    3. Predire au niveau paquet
    4. Agreger par session
    5. Retourner les verdicts mis a jour
    """
    uncertain_idx = identify_uncertain_sessions(session_probas, low, high)

    if len(uncertain_idx) == 0:
        return {}, uncertain_idx

    # Marks des sessions incertaines
    uncertain_marks = set(session_marks[uncertain_idx])

    # Filtrer paquets
    packet_subset = packet_df[packet_df["unique_link_mark"].isin(uncertain_marks)]

    if len(packet_subset) == 0:
        return {}, uncertain_idx

    # Predire
    pkt_preds, pkt_probas = predict_packets(model_rf_paquets, packet_subset, packet_features)

    # Agreger
    results = aggregate_to_session(
        pkt_preds, pkt_probas,
        packet_subset["unique_link_mark"].values,
        strategy=strategy
    )

    return results, uncertain_idx
