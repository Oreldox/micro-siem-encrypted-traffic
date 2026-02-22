"""
Composants UI partagés : CSS, cartes métriques, blocs d'explication.
"""

import streamlit as st


def inject_css():
    """Injecte le CSS custom du dashboard."""
    st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-card .value.green { color: #10b981; }
    .metric-card .value.red { color: #ef4444; }
    .metric-card .value.blue { color: #3b82f6; }
    .metric-card .value.yellow { color: #f59e0b; }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .sidebar-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 20px;
        text-align: center;
    }
    .hero-banner h1 {
        color: #e2e8f0;
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .hero-banner p {
        color: #94a3b8;
        font-size: 1rem;
    }
    .explain-box {
        background: rgba(59, 130, 246, 0.08);
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


def explain(text):
    """Affiche un bloc d'explication contextuel."""
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def render_metric_card(title, value, color="blue"):
    """Affiche une carte metrique stylisee."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value {color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)
