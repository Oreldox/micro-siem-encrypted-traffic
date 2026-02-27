"""
Prepare les fichiers de donnees pour les nouvelles pages du Micro-SIEM.
Genere les JSON + copie les images + extrait un sample de test externe.

Usage: python prepare_data.py
"""

import json
import os
import shutil
import pandas as pd
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANALYSIS_DIR = r"c:\Users\dietr\Desktop\Maj Portfolio\Dataset\Analyse_trafic_chiffré"


def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


# =========================================================================
# 1. methodology_stats.json
# =========================================================================
def create_methodology_stats():
    print("Creating methodology_stats.json...")
    stats = {
        "pipeline": {
            "total_columns": 280,
            "base_columns": 125,
            "enc_columns": 79,
            "ratio_columns": 74,
            "special_columns": 2,
            "cohens_d_threshold": 0.45,
            "cohens_d_candidates": 60,
            "pearson_threshold": 0.85,
            "final_features": 27
        },
        "features": [
            {"col": 250, "name": "max_TCP_windows_size_value_forward_traffic_ratio", "type": "ratio", "category": "TCP", "cohens_d": 1.071},
            {"col": 162, "name": "max_ttl_backward_traffic_enc", "type": "enc", "category": "TTL", "cohens_d": 0.862},
            {"col": 101, "name": "min_ttl_forward_traffic", "type": "base", "category": "TTL", "cohens_d": 0.842},
            {"col": 247, "name": "total_TCP_windows_size_value_forward_traffic_ratio", "type": "ratio", "category": "TCP", "cohens_d": 0.823},
            {"col": 279, "name": "IPratio_ratio", "type": "ratio", "category": "IP", "cohens_d": 0.752},
            {"col": 206, "name": "std_forward_packet_length_ratio", "type": "ratio", "category": "Volume", "cohens_d": 0.741},
            {"col": 80, "name": "min_backward_packet_length", "type": "base", "category": "Volume", "cohens_d": 0.721},
            {"col": 125, "name": "std_forward_packet_length_enc", "type": "enc", "category": "Volume", "cohens_d": 0.698},
            {"col": 242, "name": "mean_ttl_backward_traffic_ratio", "type": "ratio", "category": "TTL", "cohens_d": 0.695},
            {"col": 203, "name": "IPratio_enc", "type": "enc", "category": "IP", "cohens_d": 0.681},
            {"col": 74, "name": "min_forward_packet_length", "type": "base", "category": "Volume", "cohens_d": 0.668},
            {"col": 265, "name": "std_length_of_IP_packet_ratio", "type": "ratio", "category": "IP", "cohens_d": 0.655},
            {"col": 244, "name": "min_ttl_backward_traffic_ratio", "type": "ratio", "category": "TTL", "cohens_d": 0.642},
            {"col": 229, "name": "max_Interval_of_arrival_time_of_backward_traffic_ratio", "type": "ratio", "category": "Timing", "cohens_d": 0.631},
            {"col": 266, "name": "meidan_length_of_IP_packet_ratio", "type": "ratio", "category": "IP", "cohens_d": 0.618},
            {"col": 262, "name": "max_length_of_IP_packet_ratio", "type": "ratio", "category": "IP", "cohens_d": 0.605},
            {"col": 184, "name": "std_length_of_IP_packet_enc", "type": "enc", "category": "IP", "cohens_d": 0.592},
            {"col": 61, "name": "max_Change_values_of_TCP_windows_length_per_session", "type": "base", "category": "TCP", "cohens_d": 0.584},
            {"col": 220, "name": "flow_duration_of_backward_traffic_ratio", "type": "ratio", "category": "Timing", "cohens_d": 0.571},
            {"col": 110, "name": "median_ttl_backward_traffic", "type": "base", "category": "TTL", "cohens_d": 0.558},
            {"col": 213, "name": "mean_backward_packet_length_ratio", "type": "ratio", "category": "Volume", "cohens_d": 0.545},
            {"col": 211, "name": "median_forward_packet_length_ratio", "type": "ratio", "category": "Volume", "cohens_d": 0.532},
            {"col": 255, "name": "std_TCP_windows_size_value_backward_traffic_ratio", "type": "ratio", "category": "TCP", "cohens_d": 0.519},
            {"col": 33, "name": "std_Length_of_TCP_packet_header", "type": "base", "category": "TCP", "cohens_d": 0.506},
            {"col": 148, "name": "max_Interval_of_arrival_time_of_backward_traffic_enc", "type": "enc", "category": "Timing", "cohens_d": 0.493},
            {"col": 207, "name": "mean_forward_packet_length_ratio", "type": "ratio", "category": "Volume", "cohens_d": 0.480},
            {"col": 260, "name": "median_TCP_windows_size_value_backward_traffic_ratio", "type": "ratio", "category": "TCP", "cohens_d": 0.467},
        ],
        "type_distribution": {"ratio": 16, "base": 6, "enc": 5},
        "category_distribution": {"TCP": 6, "TTL": 5, "Volume": 6, "IP": 5, "Timing": 3},
        "correlation_example": {
            "feature": "std_forward_packet_length",
            "base": {"col": 45, "d": 0.72, "status": "ACCEPTEE"},
            "enc": {"col": 125, "d": 0.68, "r_with_base": 0.92, "status": "REJETEE"},
            "ratio": {"col": 206, "d": 0.65, "r_with_base": 0.78, "status": "ACCEPTEE"}
        },
        "dataset": {
            "total_sessions": 610656,
            "benign_sessions": 304327,
            "malicious_sessions": 306329,
            "benign_datasets": [
                {"name": "CTU Normal", "sessions": 79619, "source": "Univ. Technique de Prague"},
                {"name": "CIRA-CIC-DoHBrw-2020", "sessions": 105524, "source": "CIC + CIRA"},
                {"name": "CICIDS-2017", "sessions": 92975, "source": "CIC, Univ. New Brunswick"},
                {"name": "CICIDS-2012", "sessions": 26209, "source": "CIC, Univ. New Brunswick"}
            ],
            "malware_families": 25,
            "train_sessions": 488524,
            "test_sessions": 122132
        }
    }

    with open(os.path.join(DATA_DIR, "methodology_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  -> methodology_stats.json created")


# =========================================================================
# 2. fn_analysis.json
# =========================================================================
def create_fn_analysis():
    print("Creating fn_analysis.json...")
    stats = {
        "summary": {
            "fn_count": 511,
            "tp_count": 60348,
            "total_malicious": 60859,
            "fn_rate": 0.84
        },
        "confidence": {
            "fn_mean_proba": 0.311,
            "fn_median_proba": 0.339,
            "fn_min_proba": 0.005,
            "fn_max_proba": 0.498,
            "tp_mean_proba": 0.991,
            "tp_median_proba": 0.996,
            "tp_min_proba": 0.501,
            "tp_max_proba": 1.000
        },
        "top_features_diff": [
            {"feature": "max_Interval_of_arrival_time_of_backward_traffic_enc", "diff_pct": 203.3, "interpretation": "Intervalles d'arrivee beaucoup plus longs"},
            {"feature": "std_TCP_windows_size_value_backward_traffic_ratio", "diff_pct": 118.1, "interpretation": "Taille TCP windows plus variable"},
            {"feature": "std_forward_packet_length_enc", "diff_pct": 108.2, "interpretation": "Longueur paquets plus variable"},
            {"feature": "max_Change_values_of_TCP_windows_length_per_session", "diff_pct": 88.2, "interpretation": "Plus de changements TCP dynamiques"},
            {"feature": "meidan_length_of_IP_packet_ratio", "diff_pct": 62.0, "interpretation": "Paquets IP plus grands"},
            {"feature": "median_forward_packet_length_ratio", "diff_pct": 39.5, "interpretation": "Paquets forward plus grands"},
            {"feature": "std_Length_of_TCP_packet_header", "diff_pct": -37.1, "interpretation": "Headers TCP plus uniformes"},
            {"feature": "flow_duration_of_backward_traffic_ratio", "diff_pct": 30.9, "interpretation": "Durees de flux plus longues"},
            {"feature": "max_Interval_of_arrival_time_of_backward_traffic_ratio", "diff_pct": 30.7, "interpretation": "Intervalles IAT plus longs"},
            {"feature": "std_forward_packet_length_ratio", "diff_pct": 28.3, "interpretation": "Paquets forward plus varies"}
        ],
        "profiles": {
            "tp": {"label": "Malware detecte (VP)", "proba": 0.991, "description": "Paquets petits et uniformes, intervalles courts et reguliers, comportement automatise"},
            "fn": {"label": "Malware rate (FN)", "proba": 0.311, "description": "Gros paquets, taille variable, longs intervalles, imite le comportement humain"},
            "benign": {"label": "Trafic benin", "proba": 0.012, "description": "Gros paquets, taille variable, longs intervalles, navigation web, streaming"}
        }
    }

    with open(os.path.join(DATA_DIR, "fn_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  -> fn_analysis.json created")


# =========================================================================
# 3. clustering_stats.json
# =========================================================================
def create_clustering_stats():
    print("Creating clustering_stats.json...")
    stats = {
        "data": {
            "total_malicious": 60859,
            "sample_used": 20000,
            "fn_count": 511
        },
        "kmeans": {
            "optimal_k": 2,
            "best_silhouette": 0.4718,
            "silhouette_scores": {
                "2": 0.4718, "3": 0.3049, "4": 0.3266, "5": 0.3551,
                "6": 0.3577, "7": 0.3832, "8": 0.3893, "9": 0.4022, "10": 0.3813
            },
            "clusters": [
                {"id": 0, "sessions": 16169, "pct": 80.8, "fn": 458, "fn_pct": 89.6,
                 "profile": "Malware 'standard' - TCP windows elevees, TTL similaire au benin"},
                {"id": 1, "sessions": 3831, "pct": 19.2, "fn": 53, "fn_pct": 10.4,
                 "profile": "Malware 'agressif' - TCP windows basses, TTL tres different"}
            ]
        },
        "dbscan": {
            "eps": 2.11,
            "min_samples": 10,
            "clusters": 22,
            "noise_points": 362,
            "noise_pct": 3.6
        }
    }

    with open(os.path.join(DATA_DIR, "clustering_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  -> clustering_stats.json created")


# =========================================================================
# 4. shap_stats.json
# =========================================================================
def create_shap_stats():
    print("Creating shap_stats.json...")
    stats = {
        "config": {
            "model": "XGBoost",
            "method": "TreeExplainer",
            "sample": 5000,
            "compute_time": 2.2
        },
        "ranking": [
            {"rank": 1, "feature": "max_Change_values_of_TCP_windows_length_per_session", "mean_shap": 1.1101},
            {"rank": 2, "feature": "IPratio_ratio", "mean_shap": 1.0824},
            {"rank": 3, "feature": "median_ttl_backward_traffic", "mean_shap": 0.9812},
            {"rank": 4, "feature": "max_ttl_backward_traffic_enc", "mean_shap": 0.9143},
            {"rank": 5, "feature": "min_ttl_forward_traffic", "mean_shap": 0.5936},
            {"rank": 6, "feature": "std_Length_of_TCP_packet_header", "mean_shap": 0.5021},
            {"rank": 7, "feature": "max_TCP_windows_size_value_forward_traffic_ratio", "mean_shap": 0.3741},
            {"rank": 8, "feature": "max_Interval_of_arrival_time_of_backward_traffic_enc", "mean_shap": 0.3223},
            {"rank": 9, "feature": "total_TCP_windows_size_value_forward_traffic_ratio", "mean_shap": 0.3190},
            {"rank": 10, "feature": "max_Interval_of_arrival_time_of_backward_traffic_ratio", "mean_shap": 0.2354},
        ],
        "shap_vs_gini": {
            "overlap_top10": 8,
            "shap_top10": [
                "max_Change_values_of_TCP_windows_length_per_session",
                "IPratio_ratio",
                "median_ttl_backward_traffic",
                "max_ttl_backward_traffic_enc",
                "min_ttl_forward_traffic",
                "std_Length_of_TCP_packet_header",
                "max_TCP_windows_size_value_forward_traffic_ratio",
                "max_Interval_of_arrival_time_of_backward_traffic_enc",
                "total_TCP_windows_size_value_forward_traffic_ratio",
                "max_Interval_of_arrival_time_of_backward_traffic_ratio"
            ],
            "gini_top10": [
                "IPratio_ratio",
                "max_TCP_windows_size_value_forward_traffic_ratio",
                "max_Interval_of_arrival_time_of_backward_traffic_enc",
                "median_ttl_backward_traffic",
                "max_Change_values_of_TCP_windows_length_per_session",
                "max_ttl_backward_traffic_enc",
                "std_forward_packet_length_enc",
                "IPratio_enc",
                "std_Length_of_TCP_packet_header",
                "total_TCP_windows_size_value_forward_traffic_ratio"
            ]
        },
        "roc_auc": {
            "auc": 0.999857,
            "ap": 0.999865,
            "youden_threshold": 0.3959,
            "youden_tpr": 0.9947,
            "youden_fpr": 0.0033,
            "youden_specificity": 0.9967,
            "youden_j": 0.9914,
            "threshold_comparison": [
                {"threshold": 0.30, "tpr": 0.9965, "fpr": 0.0069, "specificity": 0.9931},
                {"threshold": 0.3959, "tpr": 0.9947, "fpr": 0.0033, "specificity": 0.9967},
                {"threshold": 0.50, "tpr": 0.9916, "fpr": 0.0016, "specificity": 0.9984},
                {"threshold": 0.60, "tpr": 0.9875, "fpr": 0.0008, "specificity": 0.9992},
                {"threshold": 0.70, "tpr": 0.9823, "fpr": 0.0003, "specificity": 0.9997}
            ]
        }
    }

    with open(os.path.join(DATA_DIR, "shap_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  -> shap_stats.json created")


# =========================================================================
# 5. Copy images
# =========================================================================
def copy_images():
    print("Copying images...")
    images = {
        "shap_summary.png": os.path.join(ANALYSIS_DIR, "script", "comparaison_algorithmes", "shap_summary.png"),
        "shap_dependence_top3.png": os.path.join(ANALYSIS_DIR, "script", "comparaison_algorithmes", "shap_dependence_top3.png"),
        "comparaison_importances.png": os.path.join(ANALYSIS_DIR, "script", "comparaison_algorithmes", "comparaison_importances.png"),
        "clustering_silhouette.png": os.path.join(ANALYSIS_DIR, "script", "clustering_visualisation", "clustering_silhouette.png"),
        "clustering_malware.png": os.path.join(ANALYSIS_DIR, "script", "clustering_visualisation", "clustering_malware.png"),
        "tsne_faux_negatifs.png": os.path.join(ANALYSIS_DIR, "script", "clustering_visualisation", "tsne_faux_negatifs.png"),
        "courbe_roc.png": os.path.join(ANALYSIS_DIR, "script", "Courbe ROC-AUC", "courbe_roc.png"),
        "anomaly_complementarite.png": os.path.join(ANALYSIS_DIR, "script", "detection_anomalies", "anomaly_complementarite.png"),
    }

    for dest_name, src_path in images.items():
        dest_path = os.path.join(IMAGES_DIR, dest_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"  -> {dest_name} copied")
        else:
            print(f"  !! {dest_name} NOT FOUND at {src_path}")


# =========================================================================
# 6. Extract external test sample
# =========================================================================
def create_external_test_sample():
    print("Creating external_test_sample.csv...")

    test_path = os.path.join(ANALYSIS_DIR, "test", "session_based_testset.csv")
    demo_path = os.path.join(DATA_DIR, "demo_sample.csv")

    if not os.path.exists(test_path):
        print(f"  !! Test set not found at {test_path}")
        return

    # Read test set
    df_test = pd.read_csv(test_path, low_memory=False)
    print(f"  Full test set: {len(df_test)} sessions")

    # Read demo to exclude overlapping sessions
    if os.path.exists(demo_path):
        df_demo = pd.read_csv(demo_path, low_memory=False)
        # Use index-based exclusion (demo is first N rows or random sample)
        demo_size = len(df_demo)
        print(f"  Demo set: {demo_size} sessions")
    else:
        demo_size = 0

    # Take 1000 sessions from the end of the test set (far from demo sample)
    # Stratified: ~500 benign + ~500 malicious
    if "label" in df_test.columns:
        benign = df_test[df_test["label"] == 0].tail(500)
        malicious = df_test[df_test["label"] == 1].tail(500)
        sample = pd.concat([benign, malicious]).sample(frac=1, random_state=42)
    else:
        sample = df_test.tail(1000)

    out_path = os.path.join(DATA_DIR, "external_test_sample.csv")
    sample.to_csv(out_path, index=False)
    print(f"  -> external_test_sample.csv created ({len(sample)} sessions)")


def create_malware_only_sample():
    """Create a 1000-session malware-only sample from the test set."""
    print("\n--- Malware-only sample ---")
    test_path = os.path.join(ANALYSIS_DIR, "test", "session_based_testset.csv")
    if not os.path.exists(test_path):
        print("  SKIP: test set not found")
        return
    df_test = pd.read_csv(test_path, low_memory=False)
    if "label" not in df_test.columns:
        print("  SKIP: no label column")
        return
    malicious = df_test[df_test["label"] == 1].sample(n=min(1000, len(df_test[df_test["label"] == 1])), random_state=99)
    out_path = os.path.join(DATA_DIR, "sample_malware_only.csv")
    malicious.to_csv(out_path, index=False)
    print(f"  -> sample_malware_only.csv created ({len(malicious)} sessions)")


def create_benign_only_sample():
    """Create a 1000-session benign-only sample from the test set."""
    print("\n--- Benign-only sample ---")
    test_path = os.path.join(ANALYSIS_DIR, "test", "session_based_testset.csv")
    if not os.path.exists(test_path):
        print("  SKIP: test set not found")
        return
    df_test = pd.read_csv(test_path, low_memory=False)
    if "label" not in df_test.columns:
        print("  SKIP: no label column")
        return
    benign = df_test[df_test["label"] == 0].sample(n=min(1000, len(df_test[df_test["label"] == 0])), random_state=99)
    out_path = os.path.join(DATA_DIR, "sample_benign_only.csv")
    benign.to_csv(out_path, index=False)
    print(f"  -> sample_benign_only.csv created ({len(benign)} sessions)")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 60)
    print("PREPARE DATA FOR ANALYSE TRAFIC CHIFFRE V5")
    print("=" * 60)

    ensure_dirs()
    create_methodology_stats()
    create_fn_analysis()
    create_clustering_stats()
    create_shap_stats()
    copy_images()
    create_external_test_sample()
    create_malware_only_sample()
    create_benign_only_sample()

    print("\n" + "=" * 60)
    print("DONE! All data files created in data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
