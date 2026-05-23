"""
Dashboard Data & Énergie
========================
Interface de présentation des résultats du projet :
  - Classification RP / RS
  - Prévision de courbe de consommation (CNN)
  - Génération de courbes (auto-encodeur)

Lancer avec : streamlit run app.py
"""

import streamlit as st

# ── Config de la page ───────────────────────────────────────────

st.set_page_config(
    page_title="Data & Énergie",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style global ────────────────────────────────────────────────

st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #333366;
    }

    /* Titres */
    h1, h2, h3 { font-weight: 600; }

    /* Métriques */
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar : navigation ───────────────────────────────────────

st.sidebar.markdown("## Data & Énergie")
st.sidebar.markdown("---")

PAGES = {
    "Classification RP / RS": "classification",
    "Prévision (CNN)": "prevision",
    "Génération (Auto-encodeur)": "generation",
}

choix = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Projet réalisé par Hugo BARRAT, Arthur CLAVEAU & Baptiste TARET dans le cadre du cours de Data & Énergie à l'ENPC (2025).")


# ── Routage vers la bonne vue ───────────────────────────────────

page = PAGES[choix]

if page == "classification":
    from views.classification_view import render
    render()

elif page == "prevision":
    from views.prevision_view import render
    render()

elif page == "generation":
    from views.generation_view import render
    render()