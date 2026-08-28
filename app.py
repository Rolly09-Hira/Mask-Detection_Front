# ============================================
# APPLICATION DE DETECTION DE MASQUE FACIAL
# Projet SDIA M1 - 2026
# Affichage des codes notebooks et resultats
# ============================================

import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import gdown
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Detection de Masque Facial - SDIA M1",
    page_icon="😷",
    layout="wide"
)

# ============================================
# CHARGEMENT DES MODELES ONNX
# ============================================

ANN_ONNX_ID = "1AzUUfQ3wXDyWkSTMf7RvmLZrupg1acFH"
CNN_ONNX_ID = "1poah3Kuipun9XKU-Ot4hmb5aQZo_IsF-"

def download_model_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Erreur de telechargement : {e}")
        return False

@st.cache_resource
def load_onnx_models():
    ann_path = 'models/ann_model.onnx'
    cnn_path = 'models/cnn_model.onnx'
    os.makedirs('models', exist_ok=True)

    if not os.path.exists(ann_path):
        with st.spinner('Telechargement ANN ONNX...'):
            success = download_model_from_drive(ANN_ONNX_ID, ann_path)
            if not success:
                return None, None

    if not os.path.exists(cnn_path):
        with st.spinner('Telechargement CNN ONNX...'):
            success = download_model_from_drive(CNN_ONNX_ID, cnn_path)
            if not success:
                return None, None

    try:
        ann_session = ort.InferenceSession(ann_path)
        cnn_session = ort.InferenceSession(cnn_path)
        return ann_session, cnn_session
    except Exception as e:
        st.error(f"Erreur chargement ONNX : {e}")
        return None, None

def predict_onnx(session, image):
    img_resized = cv2.resize(image, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_input})
    return result[0][0][0]

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
            "recall_without": "0.95",
            "cm": [[16, 5], [2, 38]]
        },
        "CNN": {
            "accuracy": "80.33%",
            "roc_auc": "0.929",
            "precision_with": "1.00",
            "recall_with": "0.43",
            "precision_without": "0.77",
            "recall_without": "1.00",
            "cm": [[9, 12], [0, 40]]
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

    # Métriques
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
    
    # Organisation des rôles avec noms des membres
    st.markdown("### Equipe projet")
    
    col1, col2 = st.columns(2)
    
    roles = [
        ("Role 1", "Cadrage et parties prenantes", "RATOVONJANAHARY Rojo Ny Ony Fitahiana - N°: 107I23 "),
        ("Role 2", "Donnees et modeles", "RANDRIANJAFY Nathanaël - N°079I23"),
        ("Role 3", "Evaluation et biais", "FANOMEZANIRINA Miaro Ny Anjara - N°: 197I23"),
        ("Role 4", "Explicabilite et interface", "ANDRIAMAHERIMANANA Johnson Rolly - N°: 011I23"),
        ("Role 5", "Gouvernance et documentation", "RATSIMBA Vahatriniaina - N°: 104I23")
    ]
    
    for i, (role, desc, membre) in enumerate(roles):
        if i < 3:
            with col1:
                st.markdown(f"""
                <div style="
                    background: #f0f9f0; 
                    padding: 12px 16px; 
                    border-radius: 8px; 
                    margin-bottom: 10px;
                    border-left: 4px solid #27ae60;
                ">
                    <b>{role}</b><br>
                    <span style="color: #555;">{desc}</span><br>
                    <span style="color: #888; font-size: 0.9rem;">👤 {membre}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"""
                <div style="
                    background: #f0f9f0; 
                    padding: 12px 16px; 
                    border-radius: 8px; 
                    margin-bottom: 10px;
                    border-left: 4px solid #27ae60;
                ">
                    <b>{role}</b><br>
                    <span style="color: #555;">{desc}</span><br>
                    <span style="color: #888; font-size: 0.9rem;">👤 {membre}</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Informations techniques
    col1, col2 = st.columns(2)
    with col1:
        st.info("Application hebergee sur Streamlit Cloud")
    with col2:
        st.info("Modeles ONNX charges depuis Google Drive")

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

    st.markdown("### Code de chargement des donnees")
    st.code("""
# NOTEBOOK 01 : CHARGEMENT DES DONNEES
def load_data(data_path):
    images = []
    labels = []
    categories = ['with_mask', 'without_mask']
    for category in categories:
        path = os.path.join(data_path, category)
        label = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = img / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_data('data/')
# Total : 304 images (104 avec masque, 200 sans masque)
""", language="python")

    st.markdown("---")
    st.markdown("### Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total images", RESULTS['dataset']['total'])
    with col2:
        st.metric("Avec masque", RESULTS['dataset']['with_mask'], delta="34.2%")
    with col3:
        st.metric("Sans masque", RESULTS['dataset']['without_mask'], delta="65.8%")

    # Graphique de distribution
    fig, ax = plt.subplots()
    ax.bar(['Avec masque', 'Sans masque'], [RESULTS['dataset']['with_mask'], RESULTS['dataset']['without_mask']], color=['green', 'red'])
    ax.set_ylabel('Nombre d\'images')
    ax.set_title('Distribution des classes')
    for i, v in enumerate([RESULTS['dataset']['with_mask'], RESULTS['dataset']['without_mask']]):
        ax.text(i, v + 2, str(v), ha='center')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Modeles compares")

    st.markdown("**Code d'entrainement ANN**")
    st.code("""
# NOTEBOOK 02 : ENTRAINEMENT ANN
model = Sequential([
    Input(shape=(128, 128, 3)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=40,
                    validation_data=(X_test, y_test))
""", language="python")

    st.markdown("**Code d'entrainement CNN**")
    st.code("""
# NOTEBOOK 03 : ENTRAINEMENT CNN
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=30,
                    validation_data=(X_test, y_test))
""", language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ANN - Architecture**")
        st.code("""
Flatten -> Dense(512) -> Dropout(0.5)
-> Dense(256) -> Dropout(0.3)
-> Dense(1)
Parametres : 25,297,921
""")
    with col2:
        st.markdown("**CNN - Architecture**")
        st.code("""
Conv2D(32) -> MaxPooling
Conv2D(64) -> MaxPooling
Conv2D(128) -> MaxPooling
Flatten -> Dense(256) -> Dense(128)
-> Dense(1)
Parametres : 8,710,817
""")

    st.success(f"**Modele retenu : {RESULTS['best_model']}**")

# ============================================
# PAGE 4 : EVALUATION
# ============================================
elif page == "evaluation":
    st.title("3. Evaluation et performance")

    st.markdown("### Code d'evaluation")
    st.code("""
# NOTEBOOK 04 : EVALUATION ET COMPARAISON
def evaluate_model(model, X_test, y_test, model_name):
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    return cm, auc

ann_cm, ann_auc = evaluate_model(ann_model, X_test, y_test, "ANN")
cnn_cm, cnn_auc = evaluate_model(cnn_model, X_test, y_test, "CNN")
""", language="python")

    st.markdown("---")
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
        cm_ann = RESULTS['models']['ANN']['cm']
        fig, ax = plt.subplots()
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Avec masque', 'Sans masque'],
                    yticklabels=['Avec masque', 'Sans masque'])
        ax.set_title('ANN - Matrice de confusion')
        st.pyplot(fig)
        st.caption("TN: 16 | FP: 5 | FN: 2 | TP: 38")

    with col2:
        st.markdown("**CNN**")
        cm_cnn = RESULTS['models']['CNN']['cm']
        fig, ax = plt.subplots()
        sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Oranges', ax=ax,
                    xticklabels=['Avec masque', 'Sans masque'],
                    yticklabels=['Avec masque', 'Sans masque'])
        ax.set_title('CNN - Matrice de confusion')
        st.pyplot(fig)
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

    st.markdown("### Code d'audit des biais")
    st.code("""
# NOTEBOOK 05 : AUDIT DES BIAIS
def audit_bias(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'precision_with': tp/(tp+fp) if (tp+fp)>0 else 0,
        'recall_with': tp/(tp+fn) if (tp+fn)>0 else 0,
        'precision_without': tn/(tn+fn) if (tn+fn)>0 else 0,
        'recall_without': tn/(tn+fp) if (tn+fp)>0 else 0
    }
""", language="python")

    st.markdown("---")
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

    st.markdown("### Code d'explicabilite")
    st.code("""
# NOTEBOOK 06 : EXPLICABILITE
def explain_prediction(model, image):
    prob = model.predict(image)[0][0]
    pred = 1 if prob > 0.5 else 0
    confidence = prob * 100 if pred == 1 else (1 - prob) * 100
    return {
        'prob': prob,
        'pred': pred,
        'label': 'SANS MASQUE' if pred == 1 else 'AVEC MASQUE',
        'confidence': confidence
    }
""", language="python")

    st.markdown("---")
    st.markdown("### Distribution des probabilites sur le test set")

    prob_data = pd.DataFrame({
        "Intervalle": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        "Avec masque": [8, 7, 3, 2, 1],
        "Sans masque": [1, 2, 5, 10, 22]
    })
    st.dataframe(prob_data, use_container_width=True)

    # Graphique
    fig, ax = plt.subplots()
    x = np.arange(len(prob_data))
    width = 0.35
    ax.bar(x - width/2, prob_data['Avec masque'], width, label='Avec masque', color='green')
    ax.bar(x + width/2, prob_data['Sans masque'], width, label='Sans masque', color='red')
    ax.set_xlabel('Intervalle de probabilite')
    ax.set_ylabel('Nombre d\'images')
    ax.set_title('Distribution des probabilites par classe')
    ax.set_xticks(x)
    ax.set_xticklabels(prob_data['Intervalle'])
    ax.legend()
    st.pyplot(fig)

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

    st.markdown("### Code de prediction en temps reel")
    st.code("""
# NOTEBOOK 07 : INTERFACE DE PREDICTION
def predict_image(model, image):
    img_resized = cv2.resize(image, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    prob = model.predict(img_input)[0][0]
    pred = 1 if prob > 0.5 else 0
    return prob, pred
""", language="python")

    st.markdown("---")
    st.markdown("### Testez le systeme de detection avec vos propres images")

    if ann_session is None:
        st.error("Modele non charge. Veuillez verifier les fichiers.")
        st.stop()

    model_choice = st.radio(
        "Choisir le modele",
        ["ANN (Recommandé)", "CNN"],
        horizontal=True
    )

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