"""
Page 2 : Analyse detaillee — Interpretation automatique, SHAP, radar de risque,
analyse temporelle, sessions similaires, feedback, comparaison features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import explain, render_metric_card


# =============================================================================
# FEATURE CATEGORIES — pour le radar de risque et l'interpretation
# =============================================================================

FEATURE_CATEGORIES = {
    "Timing": [
        "max_Interval_of_arrival_time_of_backward_traffic_enc",
        "max_Interval_of_arrival_time_of_backward_traffic_ratio",
        "flow_duration_of_backward_traffic_ratio",
    ],
    "Volume": [
        "min_forward_packet_length",
        "min_backward_packet_length",
        "std_forward_packet_length_enc",
        "std_forward_packet_length_ratio",
        "mean_forward_packet_length_ratio",
        "median_forward_packet_length_ratio",
        "mean_backward_packet_length_ratio",
    ],
    "IP / Ratio": [
        "IPratio_enc",
        "IPratio_ratio",
        "max_length_of_IP_packet_ratio",
        "std_length_of_IP_packet_ratio",
        # Accepte les deux orthographes (typo CIC-Darknet2020 "meidan")
        "meidan_length_of_IP_packet_ratio",
        "median_length_of_IP_packet_ratio",
        "std_length_of_IP_packet_enc",
    ],
    "Fenetre TCP": [
        "max_TCP_windows_size_value_forward_traffic_ratio",
        "total_TCP_windows_size_value_forward_traffic_ratio",
        "std_TCP_windows_size_value_backward_traffic_ratio",
        "median_TCP_windows_size_value_backward_traffic_ratio",
        "max_Change_values_of_TCP_windows_length_per_session",
        "std_Length_of_TCP_packet_header",
    ],
    "TTL": [
        "min_ttl_forward_traffic",
        "max_ttl_backward_traffic_enc",
        "median_ttl_backward_traffic",
        "mean_ttl_backward_traffic_ratio",
        "min_ttl_backward_traffic_ratio",
    ],
}

# Noms lisibles — accepte les deux orthographes
FEATURE_NAMES_FR = {
    "max_Interval_of_arrival_time_of_backward_traffic_enc": "Intervalle max entre paquets retour",
    "max_Interval_of_arrival_time_of_backward_traffic_ratio": "Ratio intervalle max retour",
    "flow_duration_of_backward_traffic_ratio": "Ratio duree du flux retour",
    "min_forward_packet_length": "Taille min paquet aller",
    "min_backward_packet_length": "Taille min paquet retour",
    "std_forward_packet_length_enc": "Ecart-type taille paquet aller",
    "std_forward_packet_length_ratio": "Ratio ecart-type taille aller",
    "mean_forward_packet_length_ratio": "Ratio taille moyenne aller",
    "median_forward_packet_length_ratio": "Ratio taille mediane aller",
    "mean_backward_packet_length_ratio": "Ratio taille moyenne retour",
    "IPratio_enc": "Ratio paquets aller/retour",
    "IPratio_ratio": "Ratio IP normalise",
    "max_length_of_IP_packet_ratio": "Ratio taille max paquet IP",
    "std_length_of_IP_packet_ratio": "Ratio ecart-type paquet IP",
    "meidan_length_of_IP_packet_ratio": "Ratio taille mediane paquet IP",
    "median_length_of_IP_packet_ratio": "Ratio taille mediane paquet IP",
    "std_length_of_IP_packet_enc": "Ecart-type taille paquet IP",
    "max_TCP_windows_size_value_forward_traffic_ratio": "Ratio fenetre TCP max aller",
    "total_TCP_windows_size_value_forward_traffic_ratio": "Ratio fenetre TCP totale aller",
    "std_TCP_windows_size_value_backward_traffic_ratio": "Ratio ecart-type fenetre TCP retour",
    "median_TCP_windows_size_value_backward_traffic_ratio": "Ratio fenetre TCP mediane retour",
    "max_Change_values_of_TCP_windows_length_per_session": "Changements fenetre TCP par session",
    "std_Length_of_TCP_packet_header": "Ecart-type en-tete TCP",
    "min_ttl_forward_traffic": "TTL minimum aller",
    "max_ttl_backward_traffic_enc": "TTL maximum retour",
    "median_ttl_backward_traffic": "TTL median retour",
    "mean_ttl_backward_traffic_ratio": "Ratio TTL moyen retour",
    "min_ttl_backward_traffic_ratio": "Ratio TTL minimum retour",
}


def render(models, session_features, config):
    st.header("Analyse detaillee d'une session")

    from src.ui_components import require_data
    if not require_data("Selectionnez une session pour voir l'analyse complete."):
        return

    df = st.session_state["data"]
    probas = st.session_state["probas"]
    preds = st.session_state["preds"]
    X = st.session_state["X"]

    # === Selection de session ===
    st.markdown("---")
    n = len(df)
    col_select, col_filter = st.columns([2, 1])
    with col_filter:
        filter_type = st.radio("Filtrer par", ["Toutes", "Alertes uniquement", "Top 50 suspectes"],
                               horizontal=True)
    with col_select:
        if filter_type == "Alertes uniquement":
            alert_indices = np.where(preds == 1)[0]
            if len(alert_indices) == 0:
                st.info("Aucune alerte avec le seuil actuel.")
                return
            idx = st.selectbox("Choisir une session", alert_indices,
                               format_func=lambda i: f"Session {i} — P={probas[i]:.4f}")
        elif filter_type == "Top 50 suspectes":
            top_indices = np.argsort(probas)[::-1][:50]
            idx = st.selectbox("Choisir une session", top_indices,
                               format_func=lambda i: f"Session {i} — P={probas[i]:.4f}")
        else:
            idx = st.number_input("Index de session", min_value=0, max_value=n-1, value=0)

    proba = probas[idx]
    verdict = "SUSPECT" if preds[idx] == 1 else "Benin"

    # =========================================================================
    # 1. VERDICT + CONFIANCE
    # =========================================================================
    st.markdown("---")
    st.subheader("Verdict")

    confidence = st.session_state.get("confidence")
    conf_val = confidence[idx] if confidence is not None else None

    n_verdict_cols = 5 if conf_val is not None else 4
    cols_v = st.columns(n_verdict_cols)

    with cols_v[0]:
        color = "red" if proba >= 0.5 else ("yellow" if proba >= 0.3 else "green")
        render_metric_card("Probabilite", f"{proba:.4f}", color)
    with cols_v[1]:
        render_metric_card("Verdict", verdict, "red" if verdict == "SUSPECT" else "green")
    with cols_v[2]:
        level = "Critique" if proba >= 0.8 else ("Eleve" if proba >= 0.5 else ("Moyen" if proba >= 0.3 else "Faible"))
        level_color = "red" if proba >= 0.8 else ("red" if proba >= 0.5 else ("yellow" if proba >= 0.3 else "green"))
        render_metric_card("Niveau de risque", level, level_color)
    with cols_v[3]:
        if "y_true" in st.session_state:
            label = "Malveillant" if st.session_state["y_true"][idx] == 1 else "Benin"
            correct = (st.session_state["y_true"][idx] == preds[idx])
            render_metric_card("Verite terrain", label, "green" if correct else "red")
        else:
            render_metric_card("Verite terrain", "N/A", "blue")
    if conf_val is not None:
        from src.confidence import confidence_label
        conf_text, conf_color = confidence_label(conf_val)
        with cols_v[4]:
            render_metric_card("Confiance", f"{conf_val:.0%} ({conf_text})", conf_color)

    # Accord inter-modeles
    probas_xgb = st.session_state.get("probas_xgb")
    if probas_xgb is not None:
        xgb_proba = probas_xgb[idx]
        rf_verdict = "malveillant" if proba >= 0.5 else "benin"
        xgb_verdict = "malveillant" if xgb_proba >= 0.5 else "benin"
        if rf_verdict == xgb_verdict:
            st.caption(f"Accord RF ({proba:.3f}) / XGBoost ({xgb_proba:.3f}) : les deux modeles concordent.")
        else:
            st.caption(
                f"Desaccord : RF dit *{rf_verdict}* ({proba:.3f}), XGBoost dit *{xgb_verdict}* ({xgb_proba:.3f}). "
                "Investigation recommandee."
            )

    # Infos reseau si disponibles
    if "unique_link_mark" in df.columns:
        mark = str(df.iloc[idx].get("unique_link_mark", ""))
        if mark and mark != "nan":
            st.caption(f"Session : {mark}")

    # =========================================================================
    # 2. INTERPRETATION AUTOMATIQUE
    # =========================================================================
    st.markdown("---")
    st.subheader("Interpretation automatique")

    feature_vals = X[idx]

    # Utiliser les stats d'entrainement si disponibles, sinon stats du dataset actuel
    from src.models import load_training_stats
    training_stats = load_training_stats(tuple(session_features))
    if training_stats is not None:
        mean_ref = training_stats["mean"]
        std_ref = training_stats["std"]
        z_source = "vs entrainement"
    else:
        mean_ref = X.mean(axis=0)
        std_ref = X.std(axis=0)
        z_source = "vs dataset actuel"

    z_scores = np.where(std_ref > 0, (feature_vals - mean_ref) / std_ref, 0)

    _render_interpretation(proba, verdict, session_features, z_scores, feature_vals)
    st.caption(f"Z-scores calcules {z_source}")

    # =========================================================================
    # 3. RADAR + TOP FEATURES
    # =========================================================================
    st.markdown("---")
    col_radar, col_top = st.columns([1, 1])

    with col_radar:
        st.subheader("Profil de risque")
        explain(
            "Ce radar montre le niveau d'anomalie de cette session dans 5 categories. "
            "Plus une branche est etendue, plus la session s'ecarte de la normale."
        )
        _render_risk_radar(session_features, z_scores)

    with col_top:
        st.subheader("Features les plus anormales")
        explain(
            "Les 7 features dont la valeur s'ecarte le plus de la moyenne. "
            "Z-score > 2 = anormalement eleve. < -2 = anormalement bas."
        )
        _render_top_features(session_features, feature_vals, mean_ref, z_scores)

    # =========================================================================
    # 4. ANALYSE COMPORTEMENTALE (temporelle + volume + TCP + TTL)
    # =========================================================================
    st.markdown("---")
    st.subheader("Analyse comportementale")
    explain(
        "Analyse multi-dimensionnelle de la session : "
        "<strong>timing</strong> (intervalles entre paquets, duree des flux), "
        "<strong>volume</strong> (tailles de paquets, asymetrie), "
        "<strong>TCP</strong> (fenetre, header), "
        "<strong>TTL</strong> (spoofing, proxy). "
        "Detecte beaconing C2, bursts, exfiltration, tunnels, outils automatises."
    )

    from src.temporal import analyze_session_timing, render_temporal_verdict
    temporal = analyze_session_timing(feature_vals, session_features, mean_ref, std_ref)
    temporal_text = render_temporal_verdict(temporal)
    st.markdown(temporal_text)

    suspicion = temporal.get("temporal_suspicion", 0)
    n_indicators = len(temporal.get("indicators", []))
    if suspicion > 0:
        col_susp1, col_susp2 = st.columns(2)
        with col_susp1:
            susp_color = "red" if suspicion >= 0.6 else "yellow"
            render_metric_card("Score de suspicion", f"{suspicion:.0%}", susp_color)
        with col_susp2:
            render_metric_card("Indicateurs detectes", f"{n_indicators}", "red" if n_indicators >= 3 else "yellow")

    # =========================================================================
    # 5. SHAP EXPLANATION — sur le RF principal (pas XGBoost)
    # =========================================================================
    st.markdown("---")
    shap_model_name = None
    shap_model = None

    # Priorite : RF (c'est lui qui fait les predictions)
    if "rf_session" in models:
        shap_model = models["rf_session"]
        shap_model_name = "Random Forest"
    elif "xgboost" in models:
        shap_model = models["xgboost"]
        shap_model_name = "XGBoost"

    if shap_model is not None:
        st.subheader(f"Decomposition SHAP ({shap_model_name})")
        explain(
            f"<strong>SHAP</strong> montre comment chaque feature influence la decision du <strong>{shap_model_name}</strong>. "
            "<span style='color:#ef4444'>Rouge</span> = pousse vers 'malveillant'. "
            "<span style='color:#3b82f6'>Bleu</span> = pousse vers 'benin'. "
            "Les features en haut sont les plus influentes."
        )

        try:
            import shap
            import matplotlib.pyplot as plt
            from src.models import load_shap_explainer

            explainer = load_shap_explainer(shap_model)
            X_session = X[idx:idx+1]
            shap_values = explainer.shap_values(X_session)

            # Gerer les formats differents (sklearn RF vs XGBoost)
            if isinstance(shap_values, list):
                # sklearn RF : [class_0, class_1], chaque = (n_samples, n_features)
                sv = shap_values[1][0]  # classe 1 (malveillant), premier sample
                base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv = shap_values[0]
                base = explainer.expected_value

            explanation = shap.Explanation(
                values=sv,
                base_values=float(base),
                data=X_session[0],
                feature_names=session_features
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close()

            # Interpretation SHAP en texte
            top_shap_idx = np.argsort(np.abs(sv))[::-1][:5]
            st.markdown(f"**Facteurs principaux de la decision ({shap_model_name}) :**")
            for i, si in enumerate(top_shap_idx, 1):
                feat_name = FEATURE_NAMES_FR.get(session_features[si], session_features[si])
                val = sv[si]
                direction = "malveillant" if val > 0 else "benin"
                st.markdown(f"{i}. **{feat_name}** : pousse vers *{direction}* (impact = {val:+.4f})")

        except Exception as e:
            st.warning(f"Erreur SHAP : {e}")
    else:
        st.info("Aucun modele disponible pour les explications SHAP.")

    # =========================================================================
    # 6. SESSIONS SIMILAIRES
    # =========================================================================
    st.markdown("---")
    st.subheader("Sessions similaires")
    explain(
        "Les 5 sessions les plus proches (distance euclidienne). "
        "Si des sessions similaires sont toutes suspectes, le verdict est plus fiable."
    )
    _render_similar_sessions(idx, X, probas, preds, session_features)

    # =========================================================================
    # 7. FEEDBACK UTILISATEUR
    # =========================================================================
    st.markdown("---")
    st.subheader("Correction manuelle")
    explain(
        "Marquez cette session si le verdict est incorrect. "
        "Vos corrections sont sauvegardees pour l'export et le calcul des metriques."
    )

    feedback_key = f"feedback_{idx}"
    corrections = st.session_state.get("user_corrections", {})
    current_correction = corrections.get(idx)

    col_fb1, col_fb2, col_fb3 = st.columns(3)
    with col_fb1:
        if st.button("Faux positif (c'est benin)", use_container_width=True,
                      disabled=(current_correction == "FP")):
            corrections[idx] = "FP"
            st.session_state["user_corrections"] = corrections
            st.rerun()
    with col_fb2:
        if st.button("Faux negatif (c'est malveillant)", use_container_width=True,
                      disabled=(current_correction == "FN")):
            corrections[idx] = "FN"
            st.session_state["user_corrections"] = corrections
            st.rerun()
    with col_fb3:
        if st.button("Annuler la correction", use_container_width=True,
                      disabled=(current_correction is None)):
            corrections.pop(idx, None)
            st.session_state["user_corrections"] = corrections
            st.rerun()

    if current_correction:
        st.caption(f"Correction enregistree : **{current_correction}** pour la session {idx}")

    n_corrections = len(corrections)
    if n_corrections > 0:
        st.caption(f"{n_corrections} correction(s) enregistree(s) au total.")

    # =========================================================================
    # 8. TABLE COMPLETE DES FEATURES
    # =========================================================================
    st.markdown("---")
    with st.expander("Tableau complet des features", expanded=False):
        explain(
            "Le <strong>Z-score</strong> mesure l'ecart par rapport a la reference : "
            "> 2 (rouge) = anormalement eleve, < -2 (bleu) = anormalement bas."
        )

        df_features = pd.DataFrame({
            "Feature": [FEATURE_NAMES_FR.get(f, f) for f in session_features],
            "Feature technique": session_features,
            "Valeur": feature_vals,
            "Moyenne ref.": mean_ref,
            "Z-score": z_scores
        }).sort_values("Z-score", key=abs, ascending=False)

        st.dataframe(
            df_features.style.background_gradient(
                subset=["Z-score"], cmap="RdBu_r", vmin=-3, vmax=3
            ).format({"Valeur": "{:.4f}", "Moyenne ref.": "{:.4f}", "Z-score": "{:.2f}"}),
            use_container_width=True, height=500
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _render_interpretation(proba, verdict, session_features, z_scores, feature_vals):
    """Genere une interpretation en texte naturel de la session."""
    anomalies = []
    for i, feat in enumerate(session_features):
        if abs(z_scores[i]) > 2:
            name = FEATURE_NAMES_FR.get(feat, feat)
            direction = "elevee" if z_scores[i] > 0 else "basse"
            anomalies.append((name, z_scores[i], direction))

    anomalies.sort(key=lambda x: abs(x[1]), reverse=True)

    if proba >= 0.8:
        intro = (
            f"**Alerte critique.** Cette session a une probabilite de **{proba:.1%}** d'etre malveillante. "
            "Le modele est tres confiant dans cette classification."
        )
    elif proba >= 0.5:
        intro = (
            f"**Session suspecte.** Probabilite de malveillance : **{proba:.1%}**. "
            "Le modele detecte des caracteristiques anormales."
        )
    elif proba >= 0.3:
        intro = (
            f"**Zone d'incertitude.** Probabilite : **{proba:.1%}**. "
            "Cette session presente des caracteristiques inhabituelles mais n'est pas clairement malveillante."
        )
    else:
        intro = (
            f"**Session probablement benigne.** Probabilite de malveillance : **{proba:.1%}**. "
            "Les caracteristiques de cette session correspondent au trafic normal."
        )

    st.markdown(intro)

    if anomalies:
        st.markdown("**Anomalies detectees :**")
        for name, zscore, direction in anomalies[:5]:
            st.markdown(f"- **{name}** : valeur anormalement {direction} (Z-score = {zscore:+.1f})")
    else:
        st.markdown("Aucune feature ne s'ecarte significativement de la normale (|Z| < 2).")

    # Analyse du type de menace potentiel
    if proba >= 0.5:
        _render_threat_hypothesis(session_features, z_scores)


def _render_threat_hypothesis(session_features, z_scores):
    """Propose une hypothese sur le type de menace."""
    cat_scores = {}
    for cat, feats in FEATURE_CATEGORIES.items():
        indices = [session_features.index(f) for f in feats if f in session_features]
        if indices:
            cat_scores[cat] = np.mean(np.abs(z_scores[indices]))

    max_cat = max(cat_scores, key=cat_scores.get) if cat_scores else None

    hypotheses = {
        "Timing": (
            "Le profil d'inter-arrivee des paquets est anormal. "
            "Hypothese : **beaconing C2** (communication reguliere avec un serveur de commande) "
            "ou **exfiltration lente** (donnees transmises a intervalles reguliers)."
        ),
        "Volume": (
            "Les tailles de paquets sont inhabituelles. "
            "Hypothese : **tunneling** (donnees encapsulees dans un protocole legitime) "
            "ou **transfert de fichiers malveillants** (paquets de taille atypique)."
        ),
        "IP / Ratio": (
            "Le ratio de paquets entre les deux directions est desequilibre. "
            "Hypothese : **exfiltration** (beaucoup plus de donnees dans un sens) "
            "ou **scan de ports** (requetes sans reponses)."
        ),
        "Fenetre TCP": (
            "Le comportement de la fenetre TCP est anormal. "
            "Hypothese : **outil automatise** (fenetre TCP fixe/predictible) "
            "ou **manipulation de protocole** (exploitation de la couche transport)."
        ),
        "TTL": (
            "Les valeurs TTL sont inhabituelles. "
            "Hypothese : **spoofing IP** (TTL incoherent avec la distance reelle) "
            "ou **proxy/VPN** (TTL modifie par un intermediaire)."
        ),
    }

    if max_cat and max_cat in hypotheses:
        st.markdown(f"**Hypothese de menace (categorie dominante : {max_cat}) :**")
        st.markdown(hypotheses[max_cat])
        st.caption("Basee sur le profil statistique — investigation manuelle requise pour confirmer.")


def _render_risk_radar(session_features, z_scores):
    """Radar chart du profil de risque par categorie."""
    categories = []
    values = []
    for cat, feats in FEATURE_CATEGORIES.items():
        indices = [session_features.index(f) for f in feats if f in session_features]
        if indices:
            score = np.mean(np.abs(z_scores[indices]))
            categories.append(cat)
            values.append(min(score, 5))

    if not categories:
        return

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed,
        fill="toself", name="Session",
        fillcolor="rgba(239, 68, 68, 0.3)",
        line=dict(color="#ef4444", width=2)
    ))
    fig.add_trace(go.Scatterpolar(
        r=[1] * len(categories_closed), theta=categories_closed,
        fill="toself", name="Normal (Z=1)",
        fillcolor="rgba(59, 130, 246, 0.1)",
        line=dict(color="#3b82f6", width=1, dash="dash")
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], showticklabels=True,
                           gridcolor="#334155", color="#94a3b8"),
            angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
            bgcolor="rgba(0,0,0,0)"
        ),
        template="plotly_dark", height=350, margin=dict(t=30, b=30, l=30, r=30),
        showlegend=True, legend=dict(y=-0.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_features(session_features, feature_vals, mean_all, z_scores):
    """Affiche les 7 features les plus anormales."""
    sorted_idx = np.argsort(np.abs(z_scores))[::-1][:7]

    for i in sorted_idx:
        name = FEATURE_NAMES_FR.get(session_features[i], session_features[i])
        val = feature_vals[i]
        avg = mean_all[i]
        z = z_scores[i]

        col_name, col_z = st.columns([3, 1])
        with col_name:
            st.markdown(f"**{name}**")
            st.caption(f"Valeur : {val:.4f} (moy : {avg:.4f})")
        with col_z:
            color = "#ef4444" if z > 2 else ("#3b82f6" if z < -2 else "#94a3b8")
            st.markdown(
                f'<div style="text-align:center;color:{color};font-size:1.2rem;font-weight:700">'
                f'Z = {z:+.1f}</div>',
                unsafe_allow_html=True
            )
        st.markdown("<hr style='margin:4px 0;border-color:#334155'>", unsafe_allow_html=True)


def _render_similar_sessions(idx, X, probas, preds, session_features):
    """Affiche les 5 sessions les plus similaires."""
    if len(X) < 2:
        st.info("Pas assez de sessions pour calculer la similarite.")
        return

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target = X_scaled[idx]
    distances = np.linalg.norm(X_scaled - target, axis=1)
    distances[idx] = np.inf

    closest = np.argsort(distances)[:5]

    data = []
    for rank, ci in enumerate(closest, 1):
        data.append({
            "Rang": rank,
            "Session": ci,
            "Distance": f"{distances[ci]:.2f}",
            "Probabilite": f"{probas[ci]:.4f}",
            "Verdict": "SUSPECT" if preds[ci] == 1 else "Benin",
        })

    df_sim = pd.DataFrame(data)
    st.dataframe(df_sim, use_container_width=True, hide_index=True)

    n_suspect = sum(1 for d in data if d["Verdict"] == "SUSPECT")
    if n_suspect >= 4:
        st.markdown(f"**{n_suspect}/5** sessions similaires sont aussi suspectes — verdict coherent.")
    elif n_suspect >= 2:
        st.markdown(f"**{n_suspect}/5** sessions similaires sont suspectes — zone mixte, investigation recommandee.")
    else:
        st.markdown(f"**{n_suspect}/5** sessions similaires sont suspectes — la majorite est benigne.")
