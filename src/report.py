"""
Export rapport : generation CSV et PDF des resultats d'analyse.
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime


def generate_csv_report(df, probas, preds, session_features,
                        confidence=None, corrections=None):
    """Genere un CSV avec toutes les colonnes + probabilite + verdict + confiance."""
    df_export = df.copy()
    df_export["probabilite_malveillance"] = probas
    df_export["verdict"] = np.where(preds == 1, "SUSPECT", "Benin")
    if confidence is not None:
        df_export["confiance"] = confidence
    if corrections:
        df_export["correction_manuelle"] = pd.Series(
            {i: c for i, c in corrections.items()}, dtype="object"
        ).reindex(df_export.index, fill_value="")
    return df_export.to_csv(index=False).encode("utf-8")


def generate_pdf_report(df, probas, preds, config, session_features,
                        y_true=None, if_preds=None,
                        confidence=None, probas_xgb=None, feature_quality=None):
    """Genere un rapport PDF complet avec metriques et graphiques."""
    from fpdf import FPDF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Page 1 : En-tete + Resume ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Micro-SIEM - Rapport d'analyse", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.cell(0, 8, f"Seuil de detection : {config['threshold']}", ln=True, align="C")
    pdf.cell(0, 8, f"Isolation Forest : {'Active' if config['use_if'] else 'Desactive'}", ln=True, align="C")
    pdf.ln(10)

    # Resume metriques
    n_total = len(df)
    n_alerts = int(preds.sum())
    n_high = int((probas >= 0.8).sum())

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Resume", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Sessions analysees : {n_total:,}", ln=True)
    pdf.cell(0, 7, f"Alertes (suspectes) : {n_alerts:,} ({100*n_alerts/n_total:.1f}%)", ln=True)
    pdf.cell(0, 7, f"Alertes critiques (P>0.8) : {n_high:,}", ln=True)

    # Qualite des donnees
    if feature_quality is not None:
        fq_avail = feature_quality.get("available", 27)
        pdf.cell(0, 7, f"Qualite des donnees : {fq_avail}/27 features", ln=True)

    # Confiance moyenne
    if confidence is not None:
        mean_conf = float(np.mean(confidence))
        pdf.cell(0, 7, f"Confiance moyenne : {mean_conf:.0%}", ln=True)

    # Accord inter-modeles
    if probas_xgb is not None:
        agreement = float(((probas >= 0.5) == (probas_xgb >= 0.5)).mean())
        pdf.cell(0, 7, f"Accord RF/XGBoost : {agreement:.1%}", ln=True)

    # Info IF
    if if_preds is not None:
        n_if = int(if_preds.sum())
        n_if_only = int(((if_preds == 1) & (preds == 0)).sum())
        pdf.cell(0, 7, f"Anomalies IF : {n_if:,} ({n_if_only:,} exclusives IF)", ln=True)

    # Metriques de performance si labels disponibles
    if y_true is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Performance du modele", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Accuracy : {100*acc:.2f}%", ln=True)
        pdf.cell(0, 7, f"Precision : {100*prec:.2f}%", ln=True)
        pdf.cell(0, 7, f"Recall : {100*rec:.2f}%", ln=True)
        pdf.cell(0, 7, f"F1-score : {f1:.4f}", ln=True)

    # --- Distribution des probabilites (graphique) ---
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Distribution des probabilites", ln=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if y_true is not None:
        ax.hist(probas[y_true == 0], bins=50, alpha=0.6, label="Benin", color="#3b82f6")
        ax.hist(probas[y_true == 1], bins=50, alpha=0.6, label="Malveillant", color="#ef4444")
        ax.legend()
    else:
        ax.hist(probas, bins=50, alpha=0.7, color="#3b82f6")
    ax.axvline(x=config["threshold"], color="red", linestyle="--", label=f"Seuil={config['threshold']}")
    ax.set_xlabel("Probabilite de malveillance")
    ax.set_ylabel("Nombre de sessions")
    ax.set_title("Distribution des scores de classification")
    plt.tight_layout()

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    img_buf.seek(0)
    pdf.image(img_buf, x=15, w=180)

    # --- Top 20 sessions suspectes ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Top 20 sessions les plus suspectes", ln=True)

    top_idx = np.argsort(probas)[::-1][:20]

    if y_true is not None:
        col_widths = [15, 30, 25, 25]
        headers = ["#", "Probabilite", "Verdict", "Label reel"]
    else:
        col_widths = [15, 35, 30]
        headers = ["#", "Probabilite", "Verdict"]

    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for rank, idx in enumerate(top_idx, 1):
        pdf.cell(col_widths[0], 6, str(rank), border=1, align="C")
        pdf.cell(col_widths[1], 6, f"{probas[idx]:.6f}", border=1, align="C")
        verdict = "SUSPECT" if preds[idx] == 1 else "Benin"
        pdf.cell(col_widths[2], 6, verdict, border=1, align="C")
        if y_true is not None:
            label = "Malveillant" if y_true[idx] == 1 else "Benin"
            pdf.cell(col_widths[3], 6, label, border=1, align="C")
        pdf.ln()

    # --- Matrice de confusion (si labels) ---
    if y_true is not None:
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Matrice de confusion", ln=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Vrais positifs (TP) : {tp:,}", ln=True)
        pdf.cell(0, 7, f"Vrais negatifs (TN) : {tn:,}", ln=True)
        pdf.cell(0, 7, f"Faux positifs (FP) : {fp:,}", ln=True)
        pdf.cell(0, 7, f"Faux negatifs (FN) : {fn:,}", ln=True)

    # --- Pied de page ---
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 7, "Genere par Micro-SIEM | Loris Dietrich", align="C")

    return bytes(pdf.output())
