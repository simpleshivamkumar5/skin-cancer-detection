
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "best_skin_cancer_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_INFO = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    color: #243247;
}
.stApp p, .stApp li, .stApp label {
    color: #243247;
}
.stApp h1, .stApp h2, .stApp h3 {
    color: #17324d;
}
.hero {
    background: linear-gradient(120deg, #17324d, #287b8e);
    padding: 28px;
    border-radius: 14px;
    color: white !important;
    margin-bottom: 22px;
}
.hero h1, .hero p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🔬 Skin Cancer AI")
st.sidebar.caption("Deep Learning Research Project")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "AI Prediction", "Model Evaluation", "About Project"]
)

st.sidebar.divider()
st.sidebar.info(
    "Educational research prototype only. "
    "It is not a medical diagnostic tool."
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Skin Cancer Detection Using Deep Learning</h1>
        <p>Skin lesion image classification using EfficientNetB0</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "This project explores transfer learning to classify "
        "dermoscopic skin lesion images into seven dataset classes."
    )

    st.subheader("Experiment 1 — Baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test images", "1,002")
    c2.metric("Test accuracy", "67.37%")
    c3.metric("Weighted F1-score", "0.70")

    st.info(
        "Image-level split. These are baseline research results, "
        "not clinical validation results."
    )

    st.subheader("Experiment 2 — Lesion-level split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test images", "992")
    c2.metric("Test accuracy", "72.48%")
    c3.metric("Macro F1-score", "0.521")

    st.metric("Melanoma recall", "44.1%")

    st.warning(
        "The experiments use different test splits, so their "
        "accuracy figures are not a direct like-for-like comparison. "
        "Melanoma recall remains limited. Never use this model to "
        "diagnose, rule out, or make treatment decisions."
    )

    st.subheader("Project highlights")
    st.markdown("""
    - EfficientNetB0 transfer learning
    - Seven skin lesion classes
    - Image upload and prediction scores
    - Evaluation results and confusion matrix
    - HAM10000 research dataset
    """)


# --------------------------------------------------
# AI PREDICTION PAGE
# --------------------------------------------------
elif page == "AI Prediction":
    st.title("Skin Lesion Image Classification")
    st.write(
        "Upload a dermoscopic image to view the model's "
        "predicted dataset class and class scores."
    )

    st.warning(
        "This tool is for educational research only. It can "
        "misclassify lesions, including melanoma. Do not use "
        "its output for medical decisions. Consult a qualified "
        "healthcare professional about concerning skin lesions."
    )

    uploaded_file = st.file_uploader(
        "Upload a skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded image",
                width=350
            )

            if st.button("Run Prediction", type="primary"):
                with st.spinner("Loading model and analysing image..."):
                    model = load_model()

                    resized = image.resize(IMG_SIZE)
                    image_array = np.asarray(
                        resized, dtype=np.float32
                    )
                    batch = np.expand_dims(image_array, axis=0)

                    predictions = model.predict(
                        batch, verbose=0
                    )[0]

                    if len(predictions) != len(CLASS_NAMES):
                        st.error(
                            "The model output does not match "
                            "the seven expected classes."
                        )
                    else:
                        best_index = int(np.argmax(predictions))
                        predicted_class = CLASS_NAMES[best_index]

                        st.subheader("Model output")
                        st.write(
                            f"Predicted dataset class: **{predicted_class}**"
                        )
                        st.write(CLASS_INFO[predicted_class])

                        st.caption(
                            "Scores are model outputs, not validated "
                            "medical probabilities or a diagnosis."
                        )

                        score_df = pd.DataFrame({
                            "Dataset class": CLASS_NAMES,
                            "Class description": [
                                CLASS_INFO[c] for c in CLASS_NAMES
                            ],
                            "Model score": predictions
                        }).sort_values(
                            "Model score", ascending=False
                        )

                        st.dataframe(
                            score_df,
                            hide_index=True,
                            width="stretch"
                        )

                        st.bar_chart(
                            score_df.set_index("Dataset class")[
                                "Model score"
                            ]
                        )

        except Exception as error:
            st.error("The image could not be processed.")
            st.code(str(error))
            st.caption(
                "Check that the model file exists and is compatible "
                "with the installed TensorFlow version."
            )


# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write(
        "Reported results from the project's two research experiments."
    )

    results = pd.DataFrame({
        "Metric": [
            "Test images",
            "Accuracy",
            "Macro F1-score",
            "Melanoma recall"
        ],
        "Experiment 1 — Baseline": [
            "1,002", "67.37%", "0.45", "0.32"
        ],
        "Experiment 2 — Lesion-level": [
            "992", "72.48%", "0.521", "0.441"
        ]
    })

    st.dataframe(results, hide_index=True, width="stretch")

    st.caption(
        "The experiments use different test splits. Their results "
        "are not a direct like-for-like comparison."
    )

    st.subheader("Experiment 2 — Confusion Matrix")

    confusion_matrix = np.array([
        [14, 4, 5, 8, 2, 3, 2],
        [4, 33, 3, 6, 7, 9, 4],
        [8, 3, 63, 6, 8, 13, 0],
        [0, 0, 0, 7, 0, 3, 0],
        [2, 2, 27, 1, 49, 29, 1],
        [0, 4, 28, 12, 49, 539, 19],
        [0, 0, 0, 1, 0, 0, 14]
    ])

    fig, ax = plt.subplots(figsize=(9, 7))
    display = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Lesion-level test confusion matrix")

    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            ax.text(
                col, row, str(confusion_matrix[row, col]),
                ha="center", va="center",
                color="white" if confusion_matrix[row, col] > 250
                else "black"
            )

    fig.colorbar(display, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "Melanoma recall in Experiment 2 was 44.1%. This limitation "
        "means the model must not be used for clinical decisions."
    )


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------
elif page == "About Project":
    st.title("About the Project")

    st.subheader("Objective")
    st.write(
        "To explore whether transfer learning can classify "
        "dermoscopic images into seven classes in the HAM10000 dataset."
    )

    st.subheader("Technology stack")
    st.markdown("""
    - Python
    - TensorFlow and Keras
    - EfficientNetB0
    - Streamlit
    - NumPy, Pandas, Matplotlib and Pillow
    """)

    st.subheader("Dataset classes")
    class_table = pd.DataFrame({
        "Class code": CLASS_NAMES,
        "Description": [
            CLASS_INFO[c] for c in CLASS_NAMES
        ]
    })
    st.dataframe(class_table, hide_index=True, width="stretch")

    st.subheader("Limitations and safety")
    st.write(
        "Dataset performance does not establish clinical safety. "
        "Performance can vary across devices, populations, image "
        "types and clinical settings. This prototype is not intended "
        "to diagnose skin cancer or replace a qualified clinician."
    )
