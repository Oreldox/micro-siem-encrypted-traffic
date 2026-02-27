"""
Aggregation de CSV paquets (Wireshark, tshark) en sessions pour le pipeline ML.

Quand un CSV contient des donnees au niveau paquet (une ligne = un paquet),
ce module les regroupe par session (5-tuple) et calcule les 27 features
attendues par le modele Random Forest.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# COLUMN MAPPING — noms de colonnes connus par format
# =============================================================================

COLUMN_ALIASES = {
    "src_ip": ["Source", "ip.src", "Src IP", "src_addr", "id.orig_h", "source_ip"],
    "dst_ip": ["Destination", "ip.dst", "Dst IP", "dst_addr", "id.resp_h", "dest_ip", "destination_ip"],
    "src_port": ["Source Port", "tcp.srcport", "udp.srcport", "Src Port", "id.orig_p", "source_port"],
    "dst_port": ["Destination Port", "tcp.dstport", "udp.dstport", "Dst Port", "id.resp_p", "dest_port", "destination_port"],
    "protocol": ["Protocol", "ip.proto", "proto", "protocol"],
    "length": ["Length", "frame.len", "ip.len", "Packet Length", "packet_length"],
    "timestamp": ["Time", "frame.time_relative", "frame.time_epoch", "ts", "timestamp", "No."],
    "ttl": ["ip.ttl", "TTL", "ttl"],
    "tcp_window": ["tcp.window_size", "tcp.window_size_value", "TCP Window", "tcp_window"],
    "tcp_header_len": ["tcp.hdr_len", "TCP Header Length", "tcp_header_len"],
    "info": ["Info", "info"],
}

# Protocoles TCP/UDP reconnus (Wireshark peut afficher "TCP", "TLSv1.2", etc.)
TCP_PROTOCOLS = {"TCP", "TLS", "TLSv1", "TLSv1.1", "TLSv1.2", "TLSv1.3",
                 "SSL", "HTTPS", "HTTP", "HTTP/2", "QUIC", "6", 6}
UDP_PROTOCOLS = {"UDP", "DNS", "DHCP", "NTP", "SSDP", "MDNS", "NBNS",
                 "QUIC", "17", 17}
ALL_NETWORK_PROTOCOLS = TCP_PROTOCOLS | UDP_PROTOCOLS


def detect_packet_csv(df):
    """Detecte si un DataFrame contient des donnees au niveau paquet.

    Retourne (is_packet_level, column_map, missing_columns) :
    - is_packet_level: True si le CSV semble etre des donnees paquets
    - column_map: dict {champ_interne: nom_colonne_dans_df}
    - missing_columns: list de champs internes non trouves
    """
    cols = set(df.columns)
    column_map = {}
    missing = []

    for field, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias in cols:
                column_map[field] = alias
                found = True
                break
        if not found:
            missing.append(field)

    # Pour etre considere paquet-level, il faut au minimum src_ip + dst_ip + length
    required = {"src_ip", "dst_ip", "length"}
    has_required = required.issubset(set(column_map.keys()))

    return has_required, column_map, missing


def aggregate_packets_to_sessions(df, column_map, target_features):
    """Agrege des paquets (une ligne = un paquet) en sessions (5-tuple).

    Utilise les memes calculs que compute_session_features() de feature_extraction.py
    pour garantir la coherence avec le pipeline PCAP.

    Retourne (df_sessions, feature_quality) :
    - df_sessions: DataFrame avec les 27 features + unique_link_mark
    - feature_quality: dict avec info sur les features calculables
    """
    # Extraire les colonnes disponibles
    src_ips = df[column_map["src_ip"]].astype(str).values
    dst_ips = df[column_map["dst_ip"]].astype(str).values
    lengths = pd.to_numeric(df[column_map["length"]], errors="coerce").fillna(0).values.astype(float)

    # Colonnes optionnelles
    has_ports = "src_port" in column_map and "dst_port" in column_map
    has_ttl = "ttl" in column_map
    has_tcp_win = "tcp_window" in column_map
    has_tcp_hdr = "tcp_header_len" in column_map
    has_protocol = "protocol" in column_map
    has_timestamp = "timestamp" in column_map

    if has_ports:
        src_ports = pd.to_numeric(df[column_map["src_port"]], errors="coerce").fillna(0).values.astype(int)
        dst_ports = pd.to_numeric(df[column_map["dst_port"]], errors="coerce").fillna(0).values.astype(int)
    else:
        src_ports = np.zeros(len(df), dtype=int)
        dst_ports = np.zeros(len(df), dtype=int)

    if has_ttl:
        ttls = pd.to_numeric(df[column_map["ttl"]], errors="coerce").fillna(0).values.astype(float)
    else:
        ttls = np.zeros(len(df))

    if has_tcp_win:
        tcp_wins = pd.to_numeric(df[column_map["tcp_window"]], errors="coerce").fillna(0).values.astype(float)
    else:
        tcp_wins = np.zeros(len(df))

    if has_tcp_hdr:
        tcp_hdrs = pd.to_numeric(df[column_map["tcp_header_len"]], errors="coerce").fillna(0).values.astype(float)
    else:
        tcp_hdrs = np.zeros(len(df))

    if has_timestamp:
        ts_raw = df[column_map["timestamp"]]
        timestamps = pd.to_numeric(ts_raw, errors="coerce")
        if timestamps.isna().all():
            # Essayer de parser comme datetime
            try:
                timestamps = pd.to_datetime(ts_raw).astype(np.int64) / 1e9
            except Exception:
                timestamps = pd.Series(range(len(df)), dtype=float)
        timestamps = timestamps.fillna(0).values.astype(float)
    else:
        timestamps = np.arange(len(df), dtype=float)

    # Filtrer par protocole si disponible
    if has_protocol:
        proto_col = df[column_map["protocol"]].astype(str).values
        mask = np.array([p.strip() in ALL_NETWORK_PROTOCOLS or
                         any(p.strip().startswith(tp) for tp in TCP_PROTOCOLS | UDP_PROTOCOLS)
                         for p in proto_col])
        if mask.sum() == 0:
            # Pas de filtrage si rien ne matche (peut-etre des numeros de protocole)
            mask = np.ones(len(df), dtype=bool)
    else:
        mask = np.ones(len(df), dtype=bool)

    # Grouper par session (5-tuple normalise)
    sessions = defaultdict(list)
    for i in range(len(df)):
        if not mask[i]:
            continue

        src_ip = src_ips[i]
        dst_ip = dst_ips[i]
        sp = src_ports[i]
        dp = dst_ports[i]

        if (src_ip, sp) < (dst_ip, dp):
            key = (src_ip, sp, dst_ip, dp)
            direction = "forward"
        else:
            key = (dst_ip, dp, src_ip, sp)
            direction = "backward"

        sessions[key].append({
            "timestamp": timestamps[i],
            "direction": direction,
            "ttl": ttls[i],
            "ip_len": lengths[i],
            "ip_header_len": 0,
            "tcp_header_len": tcp_hdrs[i],
            "tcp_win": tcp_wins[i],
            "tcp_payload_len": 0,
            "tcp_seg_len": 0,
            "flags": 0,
            "src_port": sp,
            "dst_port": dp,
        })

    if not sessions:
        return pd.DataFrame(), {"total": 0, "available": 0, "missing_fields": []}

    # Reutiliser compute_session_features de feature_extraction
    from src.feature_extraction import compute_session_features
    df_sessions = compute_session_features(dict(sessions), target_features)

    # Calculer la qualite des features
    available_fields = ["length", "timestamp"]
    if has_ports:
        available_fields.append("ports")
    if has_ttl:
        available_fields.append("ttl")
    if has_tcp_win:
        available_fields.append("tcp_window")
    if has_tcp_hdr:
        available_fields.append("tcp_header_len")

    # Estimer le nombre de features reellement calculables
    features_computable = _count_computable_features(
        has_ttl=has_ttl,
        has_tcp_win=has_tcp_win,
        has_tcp_hdr=has_tcp_hdr,
        total_features=len(target_features)
    )

    feature_quality = {
        "total": len(target_features),
        "available": features_computable,
        "available_fields": available_fields,
        "missing_fields": [f for f in ["ttl", "tcp_window", "tcp_header_len", "ports"]
                          if f not in available_fields],
        "sessions_count": len(sessions),
        "packets_used": sum(len(v) for v in sessions.values()),
    }

    return df_sessions, feature_quality


def _count_computable_features(has_ttl, has_tcp_win, has_tcp_hdr, total_features):
    """Estime combien de features sur 27 sont reellement calculables."""
    # Features toujours calculables (packet length, IP ratio, timing)
    count = 10  # packet length stats + IP ratio + inter-arrival + flow duration

    if has_ttl:
        count += 5  # TTL features (min/max/mean/median fwd/bwd)

    if has_tcp_win:
        count += 5  # TCP window features (max/total/std/median + changes)

    if has_tcp_hdr:
        count += 1  # std_Length_of_TCP_packet_header

    # IP packet length ratios (toujours calculables)
    count += 4  # max/std/median/std_enc of IP packet length

    # IPratio (toujours calculable)
    count += 2  # IPratio_enc + IPratio_ratio

    return min(count, total_features)


def get_tshark_command(filename="votre_capture.pcapng"):
    """Retourne la commande tshark pour exporter un CSV avec tous les champs necessaires."""
    return (
        f'tshark -r {filename} -T fields -E header=y -E "separator=," '
        f'-e frame.time_relative -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport '
        f'-e ip.proto -e ip.len -e ip.ttl -e tcp.window_size -e tcp.hdr_len '
        f'> export_complet.csv'
    )
