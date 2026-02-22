"""
Extraction de features depuis des fichiers PCAP et detection de format CSV.
"""

import io
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# PCAP EXTRACTION
# =============================================================================

def extract_sessions_from_pcap(pcap_bytes, max_packets=500_000):
    """Parse un PCAP avec dpkt et regroupe par session (5-tuple).

    Retourne un dict {session_key: list of packet dicts} ou chaque packet dict
    contient les champs extraits des headers IP/TCP/UDP.
    """
    import dpkt

    sessions = defaultdict(list)
    reader = dpkt.pcap.Reader(io.BytesIO(pcap_bytes))
    count = 0

    for ts, buf in reader:
        if count >= max_packets:
            break
        count += 1

        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            continue

        if not isinstance(eth.data, dpkt.ip.IP):
            continue

        ip = eth.data
        src_ip = _ip_to_str(ip.src)
        dst_ip = _ip_to_str(ip.dst)
        proto = ip.p
        ttl = ip.ttl
        ip_len = ip.len
        ip_header_len = ip.hl * 4

        src_port = 0
        dst_port = 0
        tcp_header_len = 0
        tcp_win = 0
        tcp_payload_len = 0
        tcp_seg_len = 0
        flags = 0

        if isinstance(ip.data, dpkt.tcp.TCP):
            tcp = ip.data
            src_port = tcp.sport
            dst_port = tcp.dport
            tcp_header_len = tcp.off * 4
            tcp_win = tcp.win
            tcp_payload_len = len(tcp.data)
            tcp_seg_len = len(tcp.data) + tcp_header_len
            flags = tcp.flags
        elif isinstance(ip.data, dpkt.udp.UDP):
            udp = ip.data
            src_port = udp.sport
            dst_port = udp.dport
        else:
            continue

        # Session key : 5-tuple (normalise pour bidirectionnel)
        if (src_ip, src_port) < (dst_ip, dst_port):
            key = (src_ip, src_port, dst_ip, dst_port, proto)
            direction = "forward"
        else:
            key = (dst_ip, dst_port, src_ip, src_port, proto)
            direction = "backward"

        sessions[key].append({
            "timestamp": ts,
            "direction": direction,
            "ttl": ttl,
            "ip_len": ip_len,
            "ip_header_len": ip_header_len,
            "tcp_header_len": tcp_header_len,
            "tcp_win": tcp_win,
            "tcp_payload_len": tcp_payload_len,
            "tcp_seg_len": tcp_seg_len,
            "flags": flags,
            "src_port": src_port,
            "dst_port": dst_port,
        })

    return dict(sessions)


def compute_session_features(sessions_dict, target_features):
    """Calcule les 27 features session-based a partir des paquets regroups.

    Tente de calculer chaque feature du target_features list. Les features
    non calculables sont mises a 0 avec un warning.
    """
    rows = []
    marks = []

    for session_key, packets in sessions_dict.items():
        if len(packets) < 2:
            continue

        fwd = [p for p in packets if p["direction"] == "forward"]
        bwd = [p for p in packets if p["direction"] == "backward"]

        # Extraire les arrays
        all_ttl = np.array([p["ttl"] for p in packets])
        fwd_ttl = np.array([p["ttl"] for p in fwd]) if fwd else np.array([0])
        bwd_ttl = np.array([p["ttl"] for p in bwd]) if bwd else np.array([0])

        fwd_pkt_len = np.array([p["ip_len"] for p in fwd]) if fwd else np.array([0])
        bwd_pkt_len = np.array([p["ip_len"] for p in bwd]) if bwd else np.array([0])

        fwd_tcp_win = np.array([p["tcp_win"] for p in fwd]) if fwd else np.array([0])
        bwd_tcp_win = np.array([p["tcp_win"] for p in bwd]) if bwd else np.array([0])

        all_ip_len = np.array([p["ip_len"] for p in packets])
        all_tcp_header = np.array([p["tcp_header_len"] for p in packets])

        timestamps = np.array([p["timestamp"] for p in packets])
        fwd_ts = np.array([p["timestamp"] for p in fwd]) if fwd else np.array([0])
        bwd_ts = np.array([p["timestamp"] for p in bwd]) if bwd else np.array([0])

        # TCP window changes
        all_tcp_wins = [p["tcp_win"] for p in packets]
        tcp_win_changes = sum(1 for i in range(1, len(all_tcp_wins)) if all_tcp_wins[i] != all_tcp_wins[i-1])

        # Inter-arrival times
        bwd_iat = np.diff(bwd_ts) if len(bwd_ts) > 1 else np.array([0])
        fwd_iat = np.diff(fwd_ts) if len(fwd_ts) > 1 else np.array([0])

        # IP ratio
        n_fwd = len(fwd)
        n_bwd = len(bwd)
        ip_ratio = n_fwd / max(n_bwd, 1)

        # Flow duration backward
        flow_dur_bwd = (bwd_ts[-1] - bwd_ts[0]) if len(bwd_ts) > 1 else 0

        # Compute a dict of calculable features
        # The _enc suffix = encoded (raw value), _ratio suffix = ratio
        feature_values = {
            # TTL features
            "min_ttl_forward_traffic": fwd_ttl.min(),
            "median_ttl_backward_traffic": np.median(bwd_ttl),
            "max_ttl_backward_traffic_enc": bwd_ttl.max(),
            "mean_ttl_backward_traffic_ratio": bwd_ttl.mean() / max(all_ttl.mean(), 1e-10),
            "min_ttl_backward_traffic_ratio": bwd_ttl.min() / max(all_ttl.min(), 1e-10) if all_ttl.min() > 0 else 0,

            # Packet length features
            "min_forward_packet_length": fwd_pkt_len.min(),
            "min_backward_packet_length": bwd_pkt_len.min(),
            "std_forward_packet_length_enc": fwd_pkt_len.std(),
            "std_forward_packet_length_ratio": fwd_pkt_len.std() / max(all_ip_len.std(), 1e-10),
            "mean_forward_packet_length_ratio": fwd_pkt_len.mean() / max(all_ip_len.mean(), 1e-10),
            "median_forward_packet_length_ratio": np.median(fwd_pkt_len) / max(np.median(all_ip_len), 1e-10),
            "mean_backward_packet_length_ratio": bwd_pkt_len.mean() / max(all_ip_len.mean(), 1e-10),

            # TCP window size features
            "max_TCP_windows_size_value_forward_traffic_ratio": fwd_tcp_win.max() / max(np.concatenate([fwd_tcp_win, bwd_tcp_win]).max(), 1e-10),
            "total_TCP_windows_size_value_forward_traffic_ratio": fwd_tcp_win.sum() / max(np.concatenate([fwd_tcp_win, bwd_tcp_win]).sum(), 1e-10),
            "std_TCP_windows_size_value_backward_traffic_ratio": bwd_tcp_win.std() / max(np.concatenate([fwd_tcp_win, bwd_tcp_win]).std(), 1e-10),
            "median_TCP_windows_size_value_backward_traffic_ratio": np.median(bwd_tcp_win) / max(np.median(np.concatenate([fwd_tcp_win, bwd_tcp_win])), 1e-10),
            "max_Change_values_of_TCP_windows_length_per_session": tcp_win_changes,

            # IP packet length features
            "max_length_of_IP_packet_ratio": all_ip_len.max() / max(all_ip_len.mean(), 1e-10),
            "std_length_of_IP_packet_ratio": all_ip_len.std() / max(all_ip_len.mean(), 1e-10),
            "median_length_of_IP_packet_ratio": np.median(all_ip_len) / max(all_ip_len.mean(), 1e-10),
            "std_length_of_IP_packet_enc": all_ip_len.std(),

            # IP ratio
            "IPratio_enc": ip_ratio,
            "IPratio_ratio": ip_ratio / max(ip_ratio, 1e-10),

            # TCP header
            "std_Length_of_TCP_packet_header": all_tcp_header.std(),

            # Inter-arrival / timing
            "max_Interval_of_arrival_time_of_backward_traffic_enc": bwd_iat.max(),
            "max_Interval_of_arrival_time_of_backward_traffic_ratio": bwd_iat.max() / max(np.concatenate([fwd_iat, bwd_iat]).max(), 1e-10) if np.concatenate([fwd_iat, bwd_iat]).max() > 0 else 0,
            "flow_duration_of_backward_traffic_ratio": flow_dur_bwd / max(timestamps[-1] - timestamps[0], 1e-10) if (timestamps[-1] - timestamps[0]) > 0 else 0,
        }

        # Map to target features
        row = {}
        for feat in target_features:
            row[feat] = feature_values.get(feat, 0.0)

        rows.append(row)
        marks.append(f"{session_key[0]}:{session_key[1]}-{session_key[2]}:{session_key[3]}")

    df = pd.DataFrame(rows)
    df["unique_link_mark"] = marks
    return df


# =============================================================================
# DATASET FORMAT DETECTION
# =============================================================================

DATASET_PROFILES = {
    "CIC-Darknet2020": {
        "signature_columns": [
            "max_TCP_windows_size_value_forward_traffic_ratio",
            "IPratio_ratio",
            "min_ttl_forward_traffic"
        ],
        "label_col": "label",
    },
}


def detect_dataset_format(df, target_features):
    """Detecte le format du CSV par matching de colonnes.

    Retourne (format_name, matched_features, missing_features).
    """
    cols = set(df.columns)

    # Check CIC-Darknet2020
    for name, profile in DATASET_PROFILES.items():
        sig = profile["signature_columns"]
        if all(s in cols for s in sig):
            matched = [f for f in target_features if f in cols]
            missing = [f for f in target_features if f not in cols]
            return name, matched, missing

    # Generic detection : count how many target features are present
    matched = [f for f in target_features if f in cols]
    missing = [f for f in target_features if f not in cols]

    if len(matched) > len(target_features) * 0.7:
        return "Compatible (partiel)", matched, missing
    elif len(matched) > 0:
        return "Partiellement compatible", matched, missing
    else:
        return "Format inconnu", matched, missing


def adapt_dataframe(df, target_features, column_mapping=None):
    """Adapte un DataFrame pour matcher les features attendues.

    column_mapping: dict {target_feature: source_column} pour le mapping manuel.
    Les features non mappees sont mises a 0.
    """
    result = pd.DataFrame(index=df.index)

    for feat in target_features:
        if column_mapping and feat in column_mapping:
            src_col = column_mapping[feat]
            if src_col and src_col in df.columns:
                result[feat] = df[src_col].values
            else:
                result[feat] = 0.0
        elif feat in df.columns:
            result[feat] = df[feat].values
        else:
            result[feat] = 0.0

    # Conserver les colonnes utilitaires
    for col in ["label", "unique_link_mark"]:
        if col in df.columns:
            result[col] = df[col].values

    return result


def _ip_to_str(ip_bytes):
    """Convertit des bytes IP en string."""
    return ".".join(str(b) for b in ip_bytes)
