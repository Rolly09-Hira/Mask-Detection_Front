"""
=============================================================
  Détection de Déforestation par Images Satellites
  Application Streamlit — Page d'accueil
=============================================================
"""

import streamlit as st

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Détection de Déforestation | Satellite AI",
    page_icon="🌳",
    layout="wide",
)

# ============================================================
# PAGE D'ACCUEIL
# ============================================================
st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 4rem;">🌳</div>
        <h1>ForestGuard AI</h1>
        <p style="font-size: 1.2rem; color: #555;">
            Détection automatique de la déforestation à partir d'images satellites
        </p>
        <hr style="max-width: 200px; margin: 20px auto;">
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TROIS CARTES DE PRÉSENTATION
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="
            background: #f0f9f0;
            padding: 24px 20px;
            border-radius: 12px;
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 2.5rem;">📤</div>
            <h3>Chargement d'image</h3>
            <p style="color: #555;">Importez une image satellite et obtenez une prédiction instantanée.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="
            background: #f0f9f0;
            padding: 24px 20px;
            border-radius: 12px;
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 2.5rem;">🤖</div>
            <h3>Intelligence artificielle</h3>
            <p style="color: #555;">Un modèle CNN analyse l'image pour détecter les zones déforestées.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="
            background: #f0f9f0;
            padding: 24px 20px;
            border-radius: 12px;
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 2.5rem;">📊</div>
            <h3>Visualisations</h3>
            <p style="color: #555;">Courbes d'entraînement, matrice de confusion et métriques détaillées.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MÉTRIQUES DE PERFORMANCE
# ============================================================
st.markdown("---")
st.markdown("### 📈 Performance du modèle")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Accuracy", "94.2%")
with m2:
    st.metric("Précision", "93.8%")
with m3:
    st.metric("Rappel", "92.1%")
with m4:
    st.metric("F1-score", "92.9%")
with m5:
    st.metric("AUC", "0.97")

# ============================================================
# PIED DE PAGE
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color: #888; font-size: 0.9rem; padding: 10px 0;">
        ForestGuard AI — Projet de détection de déforestation par deep learning
    </div>
    """,
    unsafe_allow_html=True,
)