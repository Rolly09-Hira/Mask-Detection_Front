# ============================================
# APPLICATION DE DETECTION DE MASQUE FACIAL
# Projet SDIA M1 - 2026
# Chargement des modeles depuis Google Drive
# Conversion en ONNX pour compatibilite Python 3.14
# ============================================

import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import gdown
import onnxruntime as ort
import tempfile

# Configuration de la page
st.set_page_config(
    page_title="Detection de Masque Facial - SDIA M1",
    page_icon="😷",
    layout="wide"
)

# ============================================
# CHARGEMENT DES MODELES DEPUIS GOOGLE DRIVE
# ============================================

# IDs des fichiers sur Google Drive
ANN_FILE_ID = "16M5LNJBhMBAZdQ-ZJW9TktJUcbOGIfHl"
CNN_FILE_ID = "1cPRSf-NDCVE3FUqybqV2nKqTp3KQnp-w"

def download_model_from_drive(file_id, output_path):
    """Telecharge un modele depuis Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Erreur de telechargement : {e}")
        return False

def convert_keras_to_onnx(keras_path, onnx_path):
    """Convertit un modele Keras en ONNX"""
    try:
        import tensorflow as tf
        from tf2onnx import convert
        
        # Charger le modele Keras
        model = tf.keras.models.load_model(keras_path)
        
        # Convertir en ONNX
        spec = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="input"),)
        model_proto, _ = convert.from_keras(model, input_signature=spec, opset=13)
        
        # Sauvegarder
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        
        return True
    except Exception as e:
        st.error(f"Erreur de conversion ONNX : {e}")
        return False

@st.cache_resource
def load_onnx_models():
    """Charge les modeles ONNX depuis Drive"""
    ann_path = 'models/ann_model.keras'
    cnn_path = 'models/cnn_model.keras'
    ann_onnx_path = 'models/ann_model.onnx'
    cnn_onnx_path = 'models/cnn_model.onnx'

    os.makedirs('models', exist_ok=True)

    # Telecharger ANN si necessaire
    if not os.path.exists(ann_path):
        with st.spinner('Telechargement du modele ANN depuis Google Drive...'):
            success = download_model_from_drive(ANN_FILE_ID, ann_path)
            if not success:
                st.error("Impossible de telecharger le modele ANN")
                return None, None

    # Telecharger CNN si necessaire
    if not os.path.exists(cnn_path):
        with st.spinner('Telechargement du modele CNN depuis Google Drive...'):
            success = download_model_from_drive(CNN_FILE_ID, cnn_path)
            if not success:
                st.error("Impossible de telecharger le modele CNN")
                return None, None

    # Convertir ANN en ONNX si necessaire
    if not os.path.exists(ann_onnx_path):
        with st.spinner('Conversion du modele ANN en ONNX...'):
            success = convert_keras_to_onnx(ann_path, ann_onnx_path)
            if not success:
                st.error("Impossible de convertir ANN en ONNX")
                return None, None

    # Convertir CNN en ONNX si necessaire
    if not os.path.exists(cnn_onnx_path):
        with st.spinner('Conversion du modele CNN en ONNX...'):
            success = convert_keras_to_onnx(cnn_path, cnn_onnx_path)
            if not success:
                st.error("Impossible de convertir CNN en ONNX")
                return None, None

    try:
        # Charger les modeles ONNX
        ann_session = ort.InferenceSession(ann_onnx_path)
        cnn_session = ort.InferenceSession(cnn_onnx_path)
        return ann_session, cnn_session
    except Exception as e:
        st.error(f"Erreur de chargement ONNX : {e}")
        return None, None

def predict_onnx(session, image):
    """Fait une prediction avec un modele ONNX"""
    # Pre-traitement
    img_resized = cv2.resize(image, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    # Prediction ONNX
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_input})
    prob = result[0][0][0]

    return prob

# Charger les modeles ONNX
ann_session, cnn_session = load_onnx_models()

# ============================================
# DONNEES PRECALCULEES
# ============================================

RESULTS = {
    "dataset": {
        "total": 304,
        "with_mask": 104,
        "without_mask": 200
    },
    "models": {
        "ANN": {
            "accuracy": "88.52%",
            "roc_auc": "0.975",
            "precision_with": "0.89",
            "recall_with": "0.76",
            "precision_without": "0.88",
            "recall_without": "0.95"
        },
        "CNN": {
            "accuracy": "80.33%",
            "roc_auc": "0.929",
            "precision_with": "1.00",
            "recall_with": "0.43",
            "precision_without": "0.77",
            "recall_without": "1.00"
        }
    },
    "best_model": "ANN"
}

# ============================================
# SIDEBAR - NAVIGATION
# ============================================

st.sidebar.title("Detection Masque")
st.sidebar.markdown("### Projet SDIA M1")

st.sidebar.markdown("---")

pages = {
    "Accueil": "accueil",
    "1. Cadrage": "cadrage",
    "2. Donnees et Modeles": "donnees",
    "3. Evaluation": "evaluation",
    "4. Audit Biais": "biais",
    "5. Explicabilite": "explicabilite",
    "6. Demo IA": "demo",
    "7. Documentation": "documentation"
}

choice = st.sidebar.radio("Aller a :", list(pages.keys()))
page = pages[choice]

st.sidebar.markdown("---")
st.sidebar.caption("Projet SDIA M1 - 2026")
st.sidebar.caption(f"Modele : {RESULTS['best_model']} | Accuracy : {RESULTS['models']['ANN']['accuracy']}")

# ============================================
# PAGE 1 : ACCUEIL
# ============================================
if page == "accueil":
    st.title("Detection de Masque Facial")
    st.markdown("### Projet SDIA M1 - Bases theoriques et ethique de l'IA")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total images", RESULTS['dataset']['total'])
    with col2:
        st.metric("Avec masque", RESULTS['dataset']['with_mask'])
    with col3:
        st.metric("Sans masque", RESULTS['dataset']['without_mask'])
    with col4:
        st.metric("Modele retenu", RESULTS['best_model'])

    st.markdown("---")
    st.markdown("### Organisation des roles")

    col1, col2 = st.columns(2)
    roles = [
        ("Role 1", "Cadrage et parties prenantes"),
        ("Role 2", "Donnees et modeles"),
        ("Role 3", "Evaluation et biais"),
        ("Role 4", "Explicabilite et interface"),
        ("Role 5", "Gouvernance et documentation")
    ]

    for i, (role, desc) in enumerate(roles):
        if i < 3:
            with col1:
                st.write(f"- **{role}** : {desc}")
        else:
            with col2:
                st.write(f"- **{role}** : {desc}")

    st.info("Application hebergee sur Streamlit Cloud")
    st.info("Les modeles sont telecharges depuis Google Drive et convertis en ONNX")

# ============================================
# PAGE 2 : CADRAGE
# ============================================
elif page == "cadrage":
    st.title("1. Cadrage et parties prenantes")

    st.markdown("### Finalite du systeme")
    st.write("""
    Detection automatique du port du masque facial pour assister la surveillance
    sanitaire dans les espaces publics (transports, commerces, etablissements scolaires).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Utilisateurs**")
        st.write("- Personnel de securite")
        st.write("- Gestionnaires d'etablissements")
        st.write("- Autorites sanitaires")

    with col2:
        st.markdown("**Personnes affectees**")
        st.write("- Usagers des espaces publics")
        st.write("- Personnel de surveillance")
        st.write("- Personnes avec dispense medicale")

    st.markdown("---")
    st.markdown("### Erreurs possibles et consequences")

    df_erreurs = pd.DataFrame({
        "Type d'erreur": ["Faux positif", "Faux negatif"],
        "Description": ["Sans masque detecte alors que masque porte", "Masque detecte alors que sans masque"],
        "Consequence": ["Sanction injustifiee", "Risque sanitaire"]
    })
    st.dataframe(df_erreurs, use_container_width=True)

# ============================================
# PAGE 3 : DONNEES ET MODELES
# ============================================
elif page == "donnees":
    st.title("2. Donnees et modeles")

    st.markdown("### Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total images", RESULTS['dataset']['total'])
    with col2:
        st.metric("Avec masque", RESULTS['dataset']['with_mask'], delta="34.2%")
    with col3:
        st.metric("Sans masque", RESULTS['dataset']['without_mask'], delta="65.8%")

    st.markdown("---")
    st.markdown("### Modeles compares")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Modele 1 : ANN (Recommandé)**")
        st.code("""
Flatten -> Dense(512) -> Dropout(0.5)
-> Dense(256) -> Dropout(0.3)
-> Dense(1)
        """)
        st.write("**Architecture :** Reseau de neurones simple")
        st.write("**Parametres :** 25,297,921")
        st.write("**Interprete :** Oui")
        st.write("**Format :** ONNX (converti)")

    with col2:
        st.markdown("**Modele 2 : CNN**")
        st.code("""
Conv2D(32) -> MaxPooling
Conv2D(64) -> MaxPooling
Conv2D(128) -> MaxPooling
Flatten -> Dense(256) -> Dense(128)
-> Dense(1)
        """)
        st.write("**Architecture :** Reseau convolutionnel")
        st.write("**Parametres :** 8,710,817")
        st.write("**Interprete :** Moins")
        st.write("**Format :** ONNX (converti)")

    st.markdown("---")
    st.success(f"**Modele retenu : {RESULTS['best_model']}** (meilleure performance sur ce dataset)")

# ============================================
# PAGE 4 : EVALUATION
# ============================================
elif page == "evaluation":
    st.title("3. Evaluation et performance")

    st.markdown("### Comparaison des modeles")

    df_comp = pd.DataFrame({
        "Modele": ["ANN", "CNN"],
        "Accuracy": [RESULTS['models']['ANN']['accuracy'], RESULTS['models']['CNN']['accuracy']],
        "ROC-AUC": [RESULTS['models']['ANN']['roc_auc'], RESULTS['models']['CNN']['roc_auc']],
        "Precision (Sans)": [RESULTS['models']['ANN']['precision_without'], RESULTS['models']['CNN']['precision_without']],
        "Recall (Sans)": [RESULTS['models']['ANN']['recall_without'], RESULTS['models']['CNN']['recall_without']]
    })
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Matrices de confusion")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ANN**")
        cm_ann = pd.DataFrame(
            [[16, 5], [2, 38]],
            index=["Reel: Avec masque", "Reel: Sans masque"],
            columns=["Pred: Avec masque", "Pred: Sans masque"]
        )
        st.dataframe(cm_ann)
        st.caption("TN: 16 | FP: 5 | FN: 2 | TP: 38")

    with col2:
        st.markdown("**CNN**")
        cm_cnn = pd.DataFrame(
            [[9, 12], [0, 40]],
            index=["Reel: Avec masque", "Reel: Sans masque"],
            columns=["Pred: Avec masque", "Pred: Sans masque"]
        )
        st.dataframe(cm_cnn)
        st.caption("TN: 9 | FP: 12 | FN: 0 | TP: 40")

    st.markdown("---")
    st.markdown("### Analyse des erreurs")
    st.write("""
    - **ANN** : 5 faux positifs, 2 faux negatifs
    - **CNN** : 12 faux positifs, 0 faux negatifs
    - Le CNN detecte tous les sans masque mais genere plus de faux positifs
    """)

# ============================================
# PAGE 5 : AUDIT BIAIS
# ============================================
elif page == "biais":
    st.title("4. Audit des biais")

    st.markdown("### Performance par classe (ANN)")

    df_metriques = pd.DataFrame({
        "Classe": ["Avec masque", "Sans masque"],
        "Precision": [RESULTS['models']['ANN']['precision_with'], RESULTS['models']['ANN']['precision_without']],
        "Recall": [RESULTS['models']['ANN']['recall_with'], RESULTS['models']['ANN']['recall_without']],
        "Faux positifs": ["5", "2"],
        "Faux negatifs": ["2", "5"]
    })
    st.dataframe(df_metriques, use_container_width=True)

    st.markdown("---")
    st.markdown("### Stress test ethique")

    scenarios = [
        "Visages avec accessoires (lunettes, chapeaux)",
        "Personnes avec barbe ou hijab",
        "Mauvais eclairage / ombres",
        "Visages d'enfants",
        "Masques de differentes couleurs",
        "Visages de profil",
        "Plusieurs personnes dans l'image"
    ]

    for s in scenarios:
        st.write(f"- {s}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Forces**")
        st.write("Bonne performance globale (88.52%)")
        st.write("Excellent ROC-AUC (0.975)")
        st.write("Faible taux de faux negatifs")

    with col2:
        st.markdown("**Faiblesses**")
        st.write("Performance asymetrique entre classes")
        st.write(f"Dataset limite ({RESULTS['dataset']['total']} images)")
        st.write("Population non diversifiee")

# ============================================
# PAGE 6 : EXPLICABILITE
# ============================================
elif page == "explicabilite":
    st.title("5. Explicabilite")

    st.markdown("### Interpretation des decisions")

    st.markdown("**Distribution des probabilites sur le test set**")

    prob_data = pd.DataFrame({
        "Intervalle": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        "Avec masque": [8, 7, 3, 2, 1],
        "Sans masque": [1, 2, 5, 10, 22]
    })
    st.dataframe(prob_data, use_container_width=True)

    st.markdown("---")
    st.markdown("### Zones d'incertitude")
    st.write("**55.7%** des predictions sont dans la zone d'incertitude (0.3 - 0.7)")
    st.write("Ces cas necessitent une verification humaine")

    st.markdown("---")
    st.markdown("### Exemples d'explication")

    examples = [
        {"label": "Avec masque", "prob": 0.0885, "conf": 91.2},
        {"label": "Sans masque", "prob": 0.7439, "conf": 74.4},
        {"label": "Avec masque", "prob": 0.1523, "conf": 84.8},
        {"label": "Sans masque", "prob": 0.8921, "conf": 89.2}
    ]

    for ex in examples:
        with st.expander(f"Exemple : {ex['label']} (Confiance : {ex['conf']:.1f}%)"):
            st.write(f"- **Probabilite brute** : {ex['prob']:.4f}")
            st.write(f"- **Prediction** : {ex['label']}")
            if ex['conf'] > 80:
                st.success("Decision fiable")
            elif ex['conf'] > 60:
                st.warning("Decision moderee - verification recommandee")
            else:
                st.error("Decision incertaine - verification requise")

# ============================================
# PAGE 7 : DEMO IA
# ============================================
elif page == "demo":
    st.title("Demo IA - Detection de Masque")

    st.markdown("### Testez le systeme de detection avec vos propres images")

    if ann_session is None:
        st.error("Modele non charge. Veuillez verifier les fichiers.")
        st.stop()

    model_choice = st.radio(
        "Choisir le modele",
        ["ANN (Recommandé)", "CNN"],
        horizontal=True
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Charger une image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargee", width=300)

        if st.button("Analyser l'image"):
            with st.spinner("Analyse en cours..."):
                img = np.array(image)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                # Prediction avec ONNX
                session = ann_session if model_choice == "ANN (Recommandé)" else cnn_session
                prob = predict_onnx(session, img)
                pred = 1 if prob > 0.5 else 0

            st.markdown("---")
            st.markdown("## Resultat de l'analyse")

            col1, col2 = st.columns(2)

            with col1:
                if pred == 0:
                    st.success("AVEC MASQUE")
                else:
                    st.error("SANS MASQUE")

                confidence = prob * 100 if pred == 1 else (1 - prob) * 100
                st.metric("Confiance", f"{confidence:.1f}%")

                if confidence > 80:
                    st.success("Confiance elevee - Detection fiable")
                elif confidence > 60:
                    st.warning("Confiance moderee - Verification recommandee")
                else:
                    st.error("Confiance faible - Verification requise")

            with col2:
                st.markdown("**Probabilites**")
                prob_with = (1 - prob) * 100
                prob_without = prob * 100

                st.progress(int(prob_with), text=f"Avec masque : {prob_with:.1f}%")
                st.progress(int(prob_without), text=f"Sans masque : {prob_without:.1f}%")

            st.markdown("---")
            st.markdown("### Details de la prediction")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probabilite brute", f"{prob:.4f}")
            with col2:
                st.metric("Seuil de decision", "0.50")
            with col3:
                st.metric("Modele utilise", model_choice)

            st.markdown("---")
            st.markdown("### Interpretation")

            if pred == 0:
                st.write("Le modele a detecte un masque sur cette image.")
                st.write(f"- Probabilite d'avoir un masque : **{prob_with:.1f}%**")
                st.write(f"- Probabilite d'etre sans masque : **{prob_without:.1f}%**")
            else:
                st.write("Le modele a detecte une absence de masque sur cette image.")
                st.write(f"- Probabilite d'etre sans masque : **{prob_without:.1f}%**")
                st.write(f"- Probabilite d'avoir un masque : **{prob_with:.1f}%**")

            st.warning("Ce systeme est un outil d'aide a la decision. Une verification humaine est recommandee pour les cas limites.")

    else:
        st.info("Veuillez charger une image pour commencer l'analyse")

# ============================================
# PAGE 8 : DOCUMENTATION
# ============================================
elif page == "documentation":
    st.title("7. Documentation")

    st.markdown("### Model Card - FaceMaskDetector")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Informations generales**")
        st.write("Nom : FaceMaskDetector")
        st.write("Version : 1.0")
        st.write("Type : Classification binaire")
        st.write(f"Modele retenu : {RESULTS['best_model']}")
        st.write("Format : ONNX (compatible Python 3.14)")

    with col2:
        st.markdown("**Performance**")
        st.write(f"Accuracy : {RESULTS['models']['ANN']['accuracy']}")
        st.write(f"ROC-AUC : {RESULTS['models']['ANN']['roc_auc']}")
        st.write(f"Precision (Sans) : {RESULTS['models']['ANN']['precision_without']}")
        st.write(f"Recall (Sans) : {RESULTS['models']['ANN']['recall_without']}")

    st.markdown("---")
    st.markdown("### Limitations")
    st.write("""
    - Dataset limite (304 images)
    - Non teste sur enfants
    - Non teste sur profils
    - Biais potentiel sur certaines populations
    """)

    st.markdown("---")
    st.markdown("### Recommandations")
    st.write("""
    1. Utiliser avec supervision humaine
    2. Ne pas utiliser comme seule preuve pour des sanctions
    3. Mettre en place un mecanisme de recours
    4. Respecter la conformite RGPD
    """)

    st.markdown("---")
    st.markdown("### Fichiers disponibles")

    reports_dir = "reports"
    if os.path.exists(reports_dir):
        files = [f for f in os.listdir(reports_dir) if f.endswith(('.md', '.txt'))]
        for f in files:
            with open(os.path.join(reports_dir, f), 'r', encoding='utf-8') as file:
                content = file.read()
            with st.expander(f"Fichier : {f}"):
                st.text(content[:1500] + "..." if len(content) > 1500 else content)
    else:
        st.warning("Dossier reports non trouve")