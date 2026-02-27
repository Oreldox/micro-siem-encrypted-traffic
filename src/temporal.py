"""
Analyse temporelle et comportementale : detection de beaconing, periodicite,
bursts, asymetrie de volume, anomalies TCP.

Fonctionne sur les features deja extraites par session
(IAT, flow duration, packet lengths, TCP window, TTL, IP ratio).
"""

import numpy as np


def analyze_session_timing(feature_vals, feature_names, mean_all, std_all):
    """Analyse le profil temporel et comportemental d'une session.

    Retourne un dict avec les indicateurs detectes.
    """
    result = {"status": "analyzed", "indicators": []}

    # Construire un dictionnaire {feature_name: {value, mean, z}}
    feat_data = {}
    for i, feat_name in enumerate(feature_names):
        val = feature_vals[i]
        avg = mean_all[i]
        std = std_all[i] if std_all[i] > 0 else 1e-10
        z = (val - avg) / std
        feat_data[feat_name] = {"value": val, "mean": avg, "std": std_all[i], "z": z}

    if not feat_data:
        result["status"] = "no_features"
        return result

    # =========================================================================
    # 1. ANALYSE TEMPORELLE (IAT + flow duration)
    # =========================================================================

    # 1a. Burst detection — IAT anormalement court
    iat_max = feat_data.get("max_Interval_of_arrival_time_of_backward_traffic_enc", {})
    if iat_max.get("z", 0) < -1.5 and iat_max.get("value", 1) < iat_max.get("mean", 1) * 0.3:
        result["indicators"].append({
            "type": "burst",
            "category": "Timing",
            "severity": "high" if iat_max.get("z", 0) < -2.5 else "medium",
            "description": (
                "Intervalle entre paquets retour anormalement court. "
                "Possible transfert de donnees en rafale (exfiltration, download malveillant)."
            ),
            "z_score": iat_max.get("z", 0),
        })

    # 1b. Beaconing C2 — IAT ratio proche de 1.0 (regulier)
    iat_ratio = feat_data.get("max_Interval_of_arrival_time_of_backward_traffic_ratio", {})
    if iat_ratio and abs(iat_ratio.get("value", 0) - 1.0) < 0.1:
        result["indicators"].append({
            "type": "beaconing",
            "category": "Timing",
            "severity": "high",
            "description": (
                "Le ratio max/moyenne de l'intervalle entre paquets est proche de 1.0 — "
                "les paquets arrivent a intervalles tres reguliers. "
                "Pattern typique d'un beaconing C2 (check-in periodique)."
            ),
            "z_score": iat_ratio.get("z", 0),
        })

    # 1c. Connexion persistante / tunnel
    flow_ratio = feat_data.get("flow_duration_of_backward_traffic_ratio", {})
    if flow_ratio and abs(flow_ratio.get("z", 0)) > 2:
        if flow_ratio["z"] > 2:
            desc = (
                "La duree du flux retour est anormalement longue par rapport au flux total. "
                "Possible connexion persistante (C2 long-polling, tunnel chiffre)."
            )
        else:
            desc = (
                "La duree du flux retour est anormalement courte. "
                "Communication unidirectionnelle (exfiltration, scan de ports)."
            )
        result["indicators"].append({
            "type": "flow_duration",
            "category": "Timing",
            "severity": "medium",
            "description": desc,
            "z_score": flow_ratio.get("z", 0),
        })

    # =========================================================================
    # 2. ANALYSE VOLUMETRIQUE (tailles de paquets + ratio)
    # =========================================================================

    # 2a. Asymetrie de volume — IP ratio tres desequilibre
    ip_ratio = feat_data.get("IPratio_enc", {})
    if ip_ratio and abs(ip_ratio.get("z", 0)) > 2.5:
        if ip_ratio["z"] > 2.5:
            desc = (
                "Le ratio paquets aller/retour est tres desequilibre (beaucoup plus d'envois). "
                "Possible exfiltration de donnees ou scan actif."
            )
            severity = "high"
        else:
            desc = (
                "Le ratio paquets aller/retour est tres faible (beaucoup plus de receptions). "
                "Possible download massif ou C2 avec gros payloads retour."
            )
            severity = "medium"
        result["indicators"].append({
            "type": "volume_asymmetry",
            "category": "Volume",
            "severity": severity,
            "description": desc,
            "z_score": ip_ratio.get("z", 0),
        })

    # 2b. Tailles de paquets uniformes — std forward tres faible
    std_fwd = feat_data.get("std_forward_packet_length_enc", {})
    if std_fwd and std_fwd.get("z", 0) < -2 and std_fwd.get("value", 1) < 10:
        result["indicators"].append({
            "type": "uniform_packets",
            "category": "Volume",
            "severity": "medium",
            "description": (
                "Les paquets aller ont des tailles tres uniformes (ecart-type quasi nul). "
                "Pattern typique d'un outil automatise (beaconing, heartbeat, keepalive malveillant)."
            ),
            "z_score": std_fwd.get("z", 0),
        })

    # 2c. Paquets tres petits — min forward/backward anormalement bas
    min_fwd = feat_data.get("min_forward_packet_length", {})
    min_bwd = feat_data.get("min_backward_packet_length", {})
    if (min_fwd and min_fwd.get("z", 0) < -2 and min_fwd.get("value", 100) < 60 and
            min_bwd and min_bwd.get("z", 0) < -2 and min_bwd.get("value", 100) < 60):
        result["indicators"].append({
            "type": "tiny_packets",
            "category": "Volume",
            "severity": "medium",
            "description": (
                "Les tailles minimales de paquets sont anormalement petites dans les deux directions. "
                "Possible scan SYN, covert channel, ou protocole minimal (C2 lightweight)."
            ),
            "z_score": min(min_fwd.get("z", 0), min_bwd.get("z", 0)),
        })

    # =========================================================================
    # 3. ANALYSE TCP (fenetre, header, changements)
    # =========================================================================

    # 3a. Fenetre TCP fixe — pas de changements
    tcp_changes = feat_data.get("max_Change_values_of_TCP_windows_length_per_session", {})
    if tcp_changes and tcp_changes.get("value", 1) == 0 and tcp_changes.get("mean", 0) > 5:
        result["indicators"].append({
            "type": "static_tcp_window",
            "category": "TCP",
            "severity": "medium",
            "description": (
                "Aucun changement de fenetre TCP pendant la session. "
                "Un navigateur ou application normale ajuste sa fenetre. "
                "Pattern d'outil automatise ou de tunnel."
            ),
            "z_score": tcp_changes.get("z", 0),
        })

    # 3b. Header TCP anormal
    std_tcp_hdr = feat_data.get("std_Length_of_TCP_packet_header", {})
    if std_tcp_hdr and abs(std_tcp_hdr.get("z", 0)) > 3:
        result["indicators"].append({
            "type": "tcp_header_anomaly",
            "category": "TCP",
            "severity": "medium",
            "description": (
                "L'ecart-type de la longueur d'en-tete TCP est anormalement "
                f"{'eleve' if std_tcp_hdr['z'] > 0 else 'bas'}. "
                "Possible manipulation de protocole ou encapsulation inhabituelle."
            ),
            "z_score": std_tcp_hdr.get("z", 0),
        })

    # =========================================================================
    # 4. ANALYSE TTL
    # =========================================================================

    # 4a. TTL anormal — spoofing ou proxy
    min_ttl_fwd = feat_data.get("min_ttl_forward_traffic", {})
    if min_ttl_fwd and abs(min_ttl_fwd.get("z", 0)) > 3:
        if min_ttl_fwd["z"] < -3:
            desc = (
                "Le TTL minimum des paquets aller est anormalement bas. "
                "Possible relais/proxy intermediaire ou TTL manipulation."
            )
        else:
            desc = (
                "Le TTL minimum des paquets aller est anormalement eleve. "
                "L'emetteur est peut-etre sur le meme reseau local ou spoofe son TTL."
            )
        result["indicators"].append({
            "type": "ttl_anomaly",
            "category": "TTL",
            "severity": "medium",
            "description": desc,
            "z_score": min_ttl_fwd.get("z", 0),
        })

    # =========================================================================
    # SCORE GLOBAL
    # =========================================================================
    if result["indicators"]:
        severity_scores = []
        for ind in result["indicators"]:
            if ind["severity"] == "high":
                severity_scores.append(1.0)
            else:
                severity_scores.append(0.5)
        # Moyenne ponderee : plus d'indicateurs = plus suspect
        base_score = max(severity_scores)
        # Bonus pour nombre d'indicateurs (plafonner a +0.3)
        bonus = min(0.1 * (len(result["indicators"]) - 1), 0.3)
        result["temporal_suspicion"] = min(base_score + bonus, 1.0)
    else:
        result["temporal_suspicion"] = 0.0

    return result


def render_temporal_verdict(analysis):
    """Retourne un texte d'interpretation de l'analyse temporelle et comportementale."""
    if analysis["status"] != "analyzed":
        return "Pas de features disponibles pour l'analyse comportementale de cette session."

    if not analysis["indicators"]:
        return (
            "**Profil comportemental normal.** Aucun pattern suspect detecte dans les "
            "intervalles entre paquets, les volumes, les fenetres TCP ou les TTL."
        )

    # Grouper par categorie
    by_category = {}
    for indicator in analysis["indicators"]:
        cat = indicator.get("category", "Autre")
        by_category.setdefault(cat, []).append(indicator)

    parts = []
    for cat, indicators in by_category.items():
        parts.append(f"**{cat} :**")
        for indicator in indicators:
            severity_icon = "🔴" if indicator["severity"] == "high" else "🟡"
            parts.append(f"{severity_icon} {indicator['description']}")

    return "\n\n".join(parts)
