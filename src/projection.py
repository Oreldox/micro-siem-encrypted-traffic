"""
Projection UMAP/t-SNE : embeddings 2D pour visualisation des clusters.
"""

import numpy as np
import plotly.graph_objects as go


def load_precomputed_embedding(path):
    """Charge un embedding pre-calcule (.npz)."""
    data = np.load(path)
    return data["embedding"], data["labels"]


def compute_projection_embedding(X, max_samples=10000):
    """Calcule une projection 2D : PCA (< 50 sessions) ou UMAP (>= 50).

    Retourne (embedding, indices, method_name).
    """
    from sklearn.preprocessing import StandardScaler

    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sub = X[indices]
    else:
        indices = np.arange(len(X))
        X_sub = X

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    n = len(X_scaled)

    if n < 50:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=min(2, X_scaled.shape[1]), random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        return embedding, indices, "PCA"
    else:
        import umap
        adjusted_neighbors = min(15, n - 1)
        reducer = umap.UMAP(
            n_components=2, n_neighbors=adjusted_neighbors, min_dist=0.1,
            random_state=42, verbose=False
        )
        embedding = reducer.fit_transform(X_scaled)
        return embedding, indices, "UMAP"


def compute_umap_embedding(X, n_neighbors=15, min_dist=0.1, max_samples=10000):
    """Deprecated — utiliser compute_projection_embedding()."""
    embedding, indices, _ = compute_projection_embedding(X, max_samples)
    return embedding, indices


def create_projection_figure(embedding, labels=None, preds=None, probas=None,
                             color_by="label", highlight_fn=False, method="UMAP"):
    """Cree un scatter plot Plotly interactif de la projection 2D."""
    fig = go.Figure()

    n = len(embedding)

    # Determiner les couleurs
    if color_by == "label" and labels is not None:
        colors = np.where(labels == 1, "#ef4444", "#3b82f6")
        hover_text = [f"Session {i}<br>Label: {'Malveillant' if labels[i]==1 else 'Benin'}"
                      for i in range(n)]
        if probas is not None:
            hover_text = [f"{h}<br>P={probas[i]:.4f}" for i, h in enumerate(hover_text)]
    elif color_by == "proba" and probas is not None:
        colors = probas
        hover_text = [f"Session {i}<br>P={probas[i]:.4f}" for i in range(n)]
    elif color_by == "verdict" and preds is not None:
        colors = np.where(preds == 1, "#ef4444", "#10b981")
        hover_text = [f"Session {i}<br>Verdict: {'SUSPECT' if preds[i]==1 else 'Benin'}"
                      for i in range(n)]
    else:
        colors = "#3b82f6"
        hover_text = [f"Session {i}" for i in range(n)]

    if color_by == "proba" and probas is not None:
        fig.add_trace(go.Scattergl(
            x=embedding[:, 0], y=embedding[:, 1],
            mode="markers",
            marker=dict(
                size=4, color=probas, colorscale="RdYlGn_r",
                cmin=0, cmax=1, colorbar=dict(title="P(malveillant)"),
                opacity=0.6
            ),
            text=hover_text, hoverinfo="text",
            name="Sessions"
        ))
    else:
        # Tracer benin et malveillant separement pour la legende
        if labels is not None and color_by == "label":
            mask_benin = labels == 0
            mask_mal = labels == 1
            fig.add_trace(go.Scattergl(
                x=embedding[mask_benin, 0], y=embedding[mask_benin, 1],
                mode="markers",
                marker=dict(size=4, color="#3b82f6", opacity=0.5),
                text=[hover_text[i] for i in range(n) if mask_benin[i]],
                hoverinfo="text", name="Benin"
            ))
            fig.add_trace(go.Scattergl(
                x=embedding[mask_mal, 0], y=embedding[mask_mal, 1],
                mode="markers",
                marker=dict(size=4, color="#ef4444", opacity=0.6),
                text=[hover_text[i] for i in range(n) if mask_mal[i]],
                hoverinfo="text", name="Malveillant"
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=embedding[:, 0], y=embedding[:, 1],
                mode="markers",
                marker=dict(size=4, color=colors if isinstance(colors, str) else colors.tolist(), opacity=0.6),
                text=hover_text, hoverinfo="text",
                name="Sessions"
            ))

    # Highlight faux negatifs
    if highlight_fn and labels is not None and preds is not None:
        fn_mask = (labels == 1) & (preds == 0)
        if fn_mask.sum() > 0:
            fig.add_trace(go.Scattergl(
                x=embedding[fn_mask, 0], y=embedding[fn_mask, 1],
                mode="markers",
                marker=dict(
                    size=10, color="rgba(0,0,0,0)",
                    line=dict(width=2, color="#f59e0b")
                ),
                text=[f"FN - Session {i}" for i in range(n) if fn_mask[i]],
                hoverinfo="text", name=f"Faux negatifs ({fn_mask.sum()})"
            ))

    ax1 = f"{method} 1"
    ax2 = f"{method} 2"
    fig.update_layout(
        xaxis_title=ax1, yaxis_title=ax2,
        template="plotly_dark", height=600,
        margin=dict(t=30, b=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig
