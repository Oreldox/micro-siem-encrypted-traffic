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
    """Parse un PCAP/PCAPNG avec dpkt et regroupe par session (5-tuple).

    Retourne un dict {session_key: list of packet dicts} ou chaque packet dict
    contient les champs extraits des headers IP/TCP/UDP.
    Detecte automatiquement le format (pcap vs pcapng) via les magic bytes.
    """
    import dpkt

    sessions = defaultdict(list)

    # Detecter le format par les magic bytes
    magic = pcap_bytes[:4]
    if magic in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4'):
        reader = dpkt.pcap.Reader(io.BytesIO(pcap_bytes))
    else:
        reader = dpkt.pcapng.Reader(io.BytesIO(pcap_bytes))
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
            # Note: "meidan" est un typo du dataset CIC-Darknet2020 d'origine, conserve pour compatibilite modele
            "meidan_length_of_IP_packet_ratio": np.median(all_ip_len) / max(all_ip_len.mean(), 1e-10),
            "median_length_of_IP_packet_ratio": np.median(all_ip_len) / max(all_ip_len.mean(), 1e-10),
            "std_length_of_IP_packet_enc": all_ip_len.std(),

            # IP ratio
            # _ratio = forward/backward packets ratio (valeur brute, pas d'auto-reference)
            "IPratio_enc": ip_ratio,
            "IPratio_ratio": ip_ratio,

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
        "type": "session_level",
    },
    "CICFlowMeter": {
        "signature_columns": ["Flow ID", "Src IP", "Dst IP", "Flow Duration"],
        "label_col": "Label",
        "type": "session_level",
    },
}

# Mapping CICFlowMeter → 27 features du modele
CICFLOWMETER_MAPPING = {
    "min_forward_packet_length": "Fwd Packet Length Min",
    "min_backward_packet_length": "Bwd Packet Length Min",
    "std_forward_packet_length_enc": "Fwd Packet Length Std",
    "mean_forward_packet_length_ratio": None,  # calcule : Fwd Mean / Total Mean
    "median_forward_packet_length_ratio": None,  # calcule
    "mean_backward_packet_length_ratio": None,  # calcule : Bwd Mean / Total Mean
    "std_forward_packet_length_ratio": None,  # calcule : Fwd Std / Total Std
    "max_length_of_IP_packet_ratio": None,  # calcule : Pkt Len Max / Pkt Len Mean
    "std_length_of_IP_packet_ratio": None,  # calcule : Pkt Len Std / Pkt Len Mean
    "median_length_of_IP_packet_ratio": None,  # calcule
    "std_length_of_IP_packet_enc": "Pkt Len Std",
    "std_Length_of_TCP_packet_header": None,  # non disponible directement
    "max_TCP_windows_size_value_forward_traffic_ratio": None,  # calcule
    "total_TCP_windows_size_value_forward_traffic_ratio": None,  # calcule
    "std_TCP_windows_size_value_backward_traffic_ratio": None,  # calcule
    "median_TCP_windows_size_value_backward_traffic_ratio": None,  # calcule
    "max_Change_values_of_TCP_windows_length_per_session": None,  # non disponible
    "min_ttl_forward_traffic": "Fwd Header Length",  # approximation
    "max_ttl_backward_traffic_enc": None,  # non disponible
    "median_ttl_backward_traffic": None,
    "mean_ttl_backward_traffic_ratio": None,
    "min_ttl_backward_traffic_ratio": None,
    "IPratio_enc": None,  # calcule : Fwd Pkts / Bwd Pkts
    "IPratio_ratio": None,  # calcule
    "max_Interval_of_arrival_time_of_backward_traffic_enc": "Bwd IAT Max",
    "max_Interval_of_arrival_time_of_backward_traffic_ratio": None,  # calcule
    "flow_duration_of_backward_traffic_ratio": None,  # calcule
}


def adapt_cicflowmeter(df, target_features):
    """Adapte un CSV CICFlowMeter pour matcher les 27 features du modele."""
    result = pd.DataFrame(index=df.index)

    # Colonnes CICFlowMeter utiles (noms standards)
    fwd_len_min = pd.to_numeric(df.get("Fwd Packet Length Min", pd.Series(dtype=float)), errors="coerce").fillna(0)
    fwd_len_max = pd.to_numeric(df.get("Fwd Packet Length Max", pd.Series(dtype=float)), errors="coerce").fillna(0)
    fwd_len_mean = pd.to_numeric(df.get("Fwd Packet Length Mean", pd.Series(dtype=float)), errors="coerce").fillna(0)
    fwd_len_std = pd.to_numeric(df.get("Fwd Packet Length Std", pd.Series(dtype=float)), errors="coerce").fillna(0)
    bwd_len_min = pd.to_numeric(df.get("Bwd Packet Length Min", pd.Series(dtype=float)), errors="coerce").fillna(0)
    bwd_len_mean = pd.to_numeric(df.get("Bwd Packet Length Mean", pd.Series(dtype=float)), errors="coerce").fillna(0)
    bwd_len_std = pd.to_numeric(df.get("Bwd Packet Length Std", pd.Series(dtype=float)), errors="coerce").fillna(0)
    pkt_len_max = pd.to_numeric(df.get("Pkt Len Max", pd.Series(dtype=float)), errors="coerce").fillna(0)
    pkt_len_mean = pd.to_numeric(df.get("Pkt Len Mean", pd.Series(dtype=float)), errors="coerce").fillna(1)
    pkt_len_std = pd.to_numeric(df.get("Pkt Len Std", pd.Series(dtype=float)), errors="coerce").fillna(0)
    fwd_pkts = pd.to_numeric(df.get("Total Fwd Packets", pd.Series(dtype=float)), errors="coerce").fillna(1)
    bwd_pkts = pd.to_numeric(df.get("Total Bwd Packets", pd.Series(dtype=float)), errors="coerce").fillna(1)
    bwd_iat_max = pd.to_numeric(df.get("Bwd IAT Max", pd.Series(dtype=float)), errors="coerce").fillna(0)
    flow_iat_max = pd.to_numeric(df.get("Flow IAT Max", pd.Series(dtype=float)), errors="coerce").fillna(1)
    flow_duration = pd.to_numeric(df.get("Flow Duration", pd.Series(dtype=float)), errors="coerce").fillna(1)
    init_fwd_win = pd.to_numeric(df.get("Init Fwd Win Byts", df.get("Init_Win_bytes_forward", pd.Series(dtype=float))), errors="coerce").fillna(0)
    init_bwd_win = pd.to_numeric(df.get("Init Bwd Win Byts", df.get("Init_Win_bytes_backward", pd.Series(dtype=float))), errors="coerce").fillna(0)

    total_len_mean = (fwd_len_mean * fwd_pkts + bwd_len_mean * bwd_pkts) / (fwd_pkts + bwd_pkts).replace(0, 1)
    total_len_std = (fwd_len_std + bwd_len_std) / 2  # approximation
    ip_ratio = fwd_pkts / bwd_pkts.replace(0, 1)

    for feat in target_features:
        if feat == "min_forward_packet_length":
            result[feat] = fwd_len_min
        elif feat == "min_backward_packet_length":
            result[feat] = bwd_len_min
        elif feat == "std_forward_packet_length_enc":
            result[feat] = fwd_len_std
        elif feat == "std_forward_packet_length_ratio":
            result[feat] = fwd_len_std / total_len_std.replace(0, 1e-10)
        elif feat == "mean_forward_packet_length_ratio":
            result[feat] = fwd_len_mean / total_len_mean.replace(0, 1e-10)
        elif feat == "median_forward_packet_length_ratio":
            result[feat] = fwd_len_mean / total_len_mean.replace(0, 1e-10)  # approx
        elif feat == "mean_backward_packet_length_ratio":
            result[feat] = bwd_len_mean / total_len_mean.replace(0, 1e-10)
        elif feat == "max_length_of_IP_packet_ratio":
            result[feat] = pkt_len_max / pkt_len_mean.replace(0, 1e-10)
        elif feat == "std_length_of_IP_packet_ratio":
            result[feat] = pkt_len_std / pkt_len_mean.replace(0, 1e-10)
        elif feat in ("median_length_of_IP_packet_ratio", "meidan_length_of_IP_packet_ratio"):
            result[feat] = pkt_len_mean / pkt_len_mean.replace(0, 1e-10)  # approx median~mean
        elif feat == "std_length_of_IP_packet_enc":
            result[feat] = pkt_len_std
        elif feat == "IPratio_enc" or feat == "IPratio_ratio":
            result[feat] = ip_ratio
        elif feat == "max_Interval_of_arrival_time_of_backward_traffic_enc":
            result[feat] = bwd_iat_max
        elif feat == "max_Interval_of_arrival_time_of_backward_traffic_ratio":
            result[feat] = bwd_iat_max / flow_iat_max.replace(0, 1e-10)
        elif feat == "flow_duration_of_backward_traffic_ratio":
            result[feat] = 0.5  # approximation sans donnees directionnelles
        elif feat == "max_TCP_windows_size_value_forward_traffic_ratio":
            total_win = init_fwd_win + init_bwd_win
            result[feat] = init_fwd_win / total_win.replace(0, 1e-10)
        elif feat == "total_TCP_windows_size_value_forward_traffic_ratio":
            total_win = init_fwd_win + init_bwd_win
            result[feat] = init_fwd_win / total_win.replace(0, 1e-10)
        elif feat == "std_TCP_windows_size_value_backward_traffic_ratio":
            result[feat] = 0.0  # non calculable depuis CICFlowMeter
        elif feat == "median_TCP_windows_size_value_backward_traffic_ratio":
            total_win = init_fwd_win + init_bwd_win
            result[feat] = init_bwd_win / total_win.replace(0, 1e-10)
        else:
            # Features non mappables → 0
            result[feat] = 0.0

    # Label si disponible
    for label_col in ["Label", "label"]:
        if label_col in df.columns:
            labels = df[label_col].astype(str).str.strip().str.lower()
            result["label"] = labels.apply(
                lambda x: 1 if x not in ("benign", "0", "normal", "legitimate") else 0
            )
            break

    # Identifiant de session
    if "Flow ID" in df.columns:
        result["unique_link_mark"] = df["Flow ID"].astype(str)
    else:
        result["unique_link_mark"] = [f"session_{i}" for i in range(len(df))]

    return result


def detect_dataset_format(df, target_features):
    """Detecte le format du CSV par matching de colonnes.

    Retourne (format_name, matched_features, missing_features, format_type).
    format_type: "session_level" ou "packet_level".
    """
    cols = set(df.columns)

    # 1. Check profils connus (session-level)
    for name, profile in DATASET_PROFILES.items():
        sig = profile["signature_columns"]
        if all(s in cols for s in sig):
            matched = [f for f in target_features if f in cols]
            missing = [f for f in target_features if f not in cols]
            return name, matched, missing, profile.get("type", "session_level")

    # 2. Check si c'est un CSV paquet-level (Wireshark, tshark)
    from src.csv_aggregation import detect_packet_csv
    is_packet, col_map, missing_fields = detect_packet_csv(df)
    if is_packet:
        return "Wireshark-Packets", [], target_features, "packet_level"

    # 3. Generic detection : count how many target features are present
    matched = [f for f in target_features if f in cols]
    missing = [f for f in target_features if f not in cols]

    if len(matched) > len(target_features) * 0.7:
        return "Compatible (partiel)", matched, missing, "session_level"
    elif len(matched) > 0:
        return "Partiellement compatible", matched, missing, "session_level"
    else:
        return "Format inconnu", matched, missing, "unknown"


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


def parse_zeek_connlog(file_content):
    """Parse un fichier Zeek conn.log (TSV avec headers #fields).

    Retourne un DataFrame pandas ou None si le format n'est pas reconnu.
    """
    lines = file_content.split("\n")
    fields = None
    separator = "\t"
    data_lines = []

    for line in lines:
        if line.startswith("#separator"):
            sep_val = line.split(" ", 1)[1].strip() if " " in line else "\\x09"
            if sep_val.startswith("\\x"):
                try:
                    separator = bytes.fromhex(sep_val[2:]).decode()
                except (ValueError, UnicodeDecodeError):
                    separator = "\t"
            elif sep_val == "\\t" or sep_val == "":
                separator = "\t"
            else:
                separator = sep_val
        elif line.startswith("#fields"):
            fields = line.split(separator)[1:] if separator in line else line.split("\t")[1:]
            fields = [f.strip() for f in fields]
        elif line.startswith("#") or not line.strip():
            continue
        else:
            data_lines.append(line)

    if not fields or not data_lines:
        return None

    rows = []
    for line in data_lines:
        parts = line.split(separator)
        if len(parts) == len(fields):
            rows.append(parts)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=fields)
    # Remplacer les valeurs Zeek "-" et "(empty)" par NaN
    df = df.replace(["-", "(empty)"], np.nan)
    return df


def adapt_zeek_connlog(df, target_features):
    """Adapte un Zeek conn.log pour matcher les 27 features du modele.

    Zeek fournit peu de features par session (compteurs de paquets/octets,
    duree). Les features non calculables sont mises a 0.
    """
    result = pd.DataFrame(index=df.index)

    # Extraire ce qui est disponible dans Zeek
    orig_pkts = pd.to_numeric(df.get("orig_pkts", pd.Series(dtype=float)), errors="coerce").fillna(1)
    resp_pkts = pd.to_numeric(df.get("resp_pkts", pd.Series(dtype=float)), errors="coerce").fillna(1)
    orig_bytes = pd.to_numeric(df.get("orig_ip_bytes", df.get("orig_bytes", pd.Series(dtype=float))), errors="coerce").fillna(0)
    resp_bytes = pd.to_numeric(df.get("resp_ip_bytes", df.get("resp_bytes", pd.Series(dtype=float))), errors="coerce").fillna(0)
    duration = pd.to_numeric(df.get("duration", pd.Series(dtype=float)), errors="coerce").fillna(0)

    ip_ratio = orig_pkts / resp_pkts.replace(0, 1)
    avg_fwd_len = orig_bytes / orig_pkts.replace(0, 1)
    avg_bwd_len = resp_bytes / resp_pkts.replace(0, 1)
    avg_total_len = (orig_bytes + resp_bytes) / (orig_pkts + resp_pkts).replace(0, 1)

    for feat in target_features:
        if feat in ("IPratio_enc", "IPratio_ratio"):
            result[feat] = ip_ratio
        elif feat == "mean_forward_packet_length_ratio":
            result[feat] = avg_fwd_len / avg_total_len.replace(0, 1e-10)
        elif feat == "mean_backward_packet_length_ratio":
            result[feat] = avg_bwd_len / avg_total_len.replace(0, 1e-10)
        elif feat == "flow_duration_of_backward_traffic_ratio":
            result[feat] = 0.5  # approximation
        else:
            result[feat] = 0.0

    # Identifiant
    if "uid" in df.columns:
        result["unique_link_mark"] = df["uid"].astype(str)
    elif "id.orig_h" in df.columns and "id.resp_h" in df.columns:
        result["unique_link_mark"] = (
            df["id.orig_h"].astype(str) + ":" +
            df.get("id.orig_p", pd.Series("0", index=df.index)).astype(str) + "-" +
            df["id.resp_h"].astype(str) + ":" +
            df.get("id.resp_p", pd.Series("0", index=df.index)).astype(str)
        )
    else:
        result["unique_link_mark"] = [f"session_{i}" for i in range(len(df))]

    return result


def _ip_to_str(ip_bytes):
    """Convertit des bytes IP en string."""
    return ".".join(str(b) for b in ip_bytes)
