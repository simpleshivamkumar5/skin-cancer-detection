
import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="DermaAI | Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_skin_cancer_model.keras"
)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_DETAILS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f8ff 0%, #eef4fa 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #12233f, #245b75);
    padding: 2.5rem;
    border-radius: 22px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { color: white; font-size: 2.6rem; }
.hero p { color: #e0edf5; font-size: 1.08rem; }
.info-card {
    background: white;
    border: 1px solid #dce6f0;
    padding: 1.2rem;
    border-radius: 15px;
    min-height: 125px;
}
.small-note {
    color: #526174;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔬 DermaAI")
st.sidebar.caption("Skin Lesion Classification Research")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "AI Prediction", "Model Evaluation", "About Project"]
)

st.sidebar.divider()
st.sidebar.caption("B.Sc. IT Mini Project")
st.sidebar.caption("Deep Learning • EfficientNetB0")

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>DermaAI</h1>
        <p>Skin Lesion Classification Using Deep Learning</p>
        <p>
        An educational research prototype using transfer learning
        and the HAM10000 dermoscopic image dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore the project")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
        <h3>📤 Image Upload</h3>
        Upload a dermoscopic image for model analysis.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
        <h3>🧠 AI Classification</h3>
        Explore predictions across seven lesion categories.
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
        <h3>📊 Evaluation</h3>
        Review test metrics and model limitations.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Baseline experiment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test images", "1,002")
    c2.metric("Test accuracy", "67.37%")
    c3.metric("Weighted F1-score", "0.70")

    st.info(
        "These are results from our current baseline experiment. "
        "They are not clinical validation results."
    )

    st.warning(
        "Research prototype only: this system cannot diagnose skin cancer "
        "or rule it out. Seek assessment from a qualified dermatologist "
        "for concerning or changing skin lesions."
    )

# ---------------- PREDICTION ----------------
elif page == "AI Prediction":
    st.title("Skin Lesion Image Analysis")
    st.write(
        "Upload a dermoscopic image to see the model's predicted class "
        "and its relative output scores."
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.image(
                    image,
                    caption="Uploaded image",
                    use_container_width=True
                )

            with col2:
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Running model inference..."):
                        model = load_model()

                        resized = image.resize((224, 224))
                        array = np.asarray(
                            resized, dtype=np.float32
                        )
                        batch = np.expand_dims(array, axis=0)

                        scores = model.predict(batch, verbose=0)[0]
                        predicted_index = int(np.argmax(scores))
                        predicted_class = CLASS_NAMES[predicted_index]

                        st.subheader("Model output")
                        st.write(
                            "**Predicted category:**",
                            predicted_class
                        )
                        st.write(
                            CLASS_DETAILS[predicted_class]
                        )

                        st.write(
                            "**Highest model score:** "
                            f"{scores[predicted_index] * 100:.2f}%"
                        )

                        st.caption(
                            "This score is a model output, not a calibrated "
                            "probability of disease or a measure of diagnostic certainty."
                        )

                        score_df = pd.DataFrame({
                            "Category": CLASS_NAMES,
                            "Description": [
                                CLASS_DETAILS[name]
                                for name in CLASS_NAMES
                            ],
                            "Model score (%)": np.round(
                                scores * 100, 2
                            )
                        }).sort_values(
                            "Model score (%)", ascending=False
                        )

                        st.subheader("Scores across all categories")
                        st.dataframe(
                            score_df,
                            hide_index=True,
                            use_container_width=True
                        )

            st.warning(
                "This experimental model may misclassify lesions, "
                "including melanoma. Do not use its output to decide "
                "whether to seek medical care."
            )

        except Exception as error:
            st.error("Could not process the image.")
            st.exception(error)

# ---------------- EVALUATION ----------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Baseline EfficientNetB0 experiment on the held-out test set.")

    st.metric("Test accuracy", "67.37%")
    st.metric("Weighted F1-score", "0.70")
    st.metric("Macro F1-score", "0.45")

    st.subheader("Classification report")

    report = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Precision": [0.27, 0.44, 0.54, 0.09, 0.34, 0.91, 0.46],
        "Recall":    [0.47, 0.42, 0.39, 0.73, 0.32, 0.80, 0.86],
        "F1-score":  [0.34, 0.43, 0.45, 0.16, 0.33, 0.85, 0.60],
        "Support":   [32, 52, 110, 11, 112, 671, 14]
    })

    st.dataframe(report, hide_index=True, use_container_width=True)

    st.subheader("Confusion matrix")

    cm = np.array([
        [15, 4, 1, 8, 3, 1, 0],
        [10, 22, 4, 9, 1, 5, 1],
        [14, 6, 43, 15, 14, 17, 1],
        [0, 1, 0, 8, 0, 2, 0],
        [14, 7, 13, 13, 36, 28, 1],
        [2, 9, 19, 38, 53, 539, 11],
        [0, 1, 0, 0, 0, 1, 12]
    ])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set(
        xticks=np.arange(7),
        yticks=np.arange(7),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test-set confusion matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "Class-level performance is uneven, particularly for melanoma "
        "and minority categories. The model is not suitable for clinical use."
    )

# ---------------- ABOUT ----------------
elif page == "About Project":
    st.title("About the Project")

    st.subheader("Objective")
    st.write(
        "To develop an educational image-classification prototype "
        "using deep learning and transfer learning for seven skin-lesion "
        "categories in the HAM10000 dataset."
    )

    st.subheader("Dataset")
    st.write(
        "HAM10000 contains 10,015 dermoscopic images across seven "
        "diagnostic categories: akiec, bcc, bkl, df, mel, nv and vasc."
    )

    st.subheader("Technology stack")
    st.markdown("""
    - Python
    - TensorFlow / Keras
    - EfficientNetB0 transfer learning
    - NumPy and Pandas
    - Matplotlib
    - Streamlit
    - Google Colab
    """)

    st.subheader("Current baseline results")
    st.write(
        "Test accuracy: 67.37%. Weighted F1-score: approximately 0.70. "
        "Macro F1-score: approximately 0.45."
    )

    st.subheader("Limitations and responsible use")
    st.markdown("""
    - Performance differs substantially across categories.
    - Dataset performance does not establish real-world clinical safety.
    - The model may produce incorrect or misleading predictions.
    - Do not use the website for diagnosis, treatment, or triage.
    - Medical concerns should be assessed by a qualified clinician.
    """)

    st.subheader("Project team")
    st.write("Team leader: Shivam Kumar")
    st.write("Team members: Aryan Rawat, Aditya Chauhan and Zaid")

st.sidebar.divider()
st.sidebar.caption("Educational prototype — not a medical device.")
