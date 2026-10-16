
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

MODEL_PATH = Path(__file__).parent / "experiment4_finetuned_v2.keras"
IMG_SIZE = (224, 224)
VALIDATION_ACCURACY = "76.85%"

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
/* Overall page */
.stApp {
    background: #f3f7fc !important;
    color: #172b4d !important;
}

/* Main content */
[data-testid="stAppViewContainer"] {
    background: #f3f7fc !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label,
[data-testid="stMain"] span {
    color: #263b57;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #12304a !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #12304a, #087e8b);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(18, 48, 74, 0.15);
}

.hero h1,
.hero p {
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce6f1;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(25, 55, 90, 0.07);
}

[data-testid="stMetricLabel"] p {
    color: #536780 !important;
}

[data-testid="stMetricValue"] {
    color: #087e8b !important;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12304a !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #087e8b, #10a3a3);
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background: #075e68;
    border: none;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #9bb8cc;
    border-radius: 12px;
    padding: 12px;
}

/* Make the hero title and subtitle clearly visible */
.hero,
.hero h1,
.hero h2,
.hero p {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Keep the hero title bright and readable */
.hero h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* Improve visibility of Streamlit toolbar icons */
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] button svg {
    color: #17324d !important;
    fill: #17324d !important;
    opacity: 1 !important;
}

/* Keep toolbar icons visible against the light page */
[data-testid="stHeader"] {
    background: #f3f7fc !important;
}

/* Improve visibility of the top-right menu */
[data-testid="stMainMenu"] button,
[data-testid="stMainMenu"] button svg {
    color: #17324d !important;
    fill: #17324d !important;
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

    st.subheader("Current Model — Experiment 4 V2")
    st.metric("Validation accuracy", VALIDATION_ACCURACY)
    st.caption(
        "Validation result on the project validation split. This is not test accuracy, "
        "and it does not represent the probability that an individual prediction is correct."
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
        "Reported results from the project's research experiments."
    )

    st.subheader("Current Model — Experiment 4 V2")
    st.metric("Validation accuracy", VALIDATION_ACCURACY)
    st.caption(
        "This is validation accuracy, not test accuracy. The latest model has not been "
        "confirmed with a final held-out test evaluation in this app."
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
    st.info(
        "Experiment 4 V2 validation accuracy: 76.85%. This is reported separately "
        "because validation accuracy must not be presented as test accuracy."
    )

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


# ----------------------------- ABOUT PROJECT -----------------------------
elif page == "About Project":

    hero(
        "PROJECT INFORMATION",
        "About DermaAI",
        "Understanding the science, dataset, methodology and technology "
        "behind our skin-lesion classification research prototype.",
        "B.SC. IT MINI PROJECT · DEEP LEARNING",
    )

    # 1. Project overview
    section_label("01 · PROJECT OVERVIEW")
    st.markdown("""
    <div class="panel">
        <h3>Bringing deep learning into medical-image research</h3>
        <p>
        <b>DermaAI — Skin Lesion Classification Using Deep Learning</b>
        is an educational research project that explores how artificial
        intelligence can classify dermoscopic images into seven categories.
        </p>
        <p>
        The project uses the HAM10000 dataset and an EfficientNetB0-based
        transfer-learning approach. A Streamlit web application connects
        the trained model with an interactive interface where users can
        upload an image and examine the model's output scores.
        </p>
        <p>
        The purpose is to demonstrate the complete machine-learning workflow,
        from dataset preparation and model evaluation to web deployment.
        This is a research prototype, not a clinically validated diagnostic
        system.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # 2. Objectives
    section_label("02 · PROJECT OBJECTIVES")
    st.subheader("What are we trying to achieve?")

    objectives = [
        ("01", "Image classification",
         "Explore the classification of dermoscopic images into seven dataset categories."),
        ("02", "Transfer learning",
         "Investigate EfficientNetB0 as a pretrained image-classification architecture."),
        ("03", "Data preparation",
         "Organize image metadata, labels and image dimensions for model training."),
        ("04", "Model evaluation",
         "Study accuracy, precision, recall, F1-score and confusion-matrix results."),
        ("05", "Web integration",
         "Connect the trained model to an accessible Streamlit interface."),
        ("06", "Responsible AI",
         "Understand class imbalance, model errors and the limitations of medical-image AI."),
    ]

    for start in range(0, len(objectives), 3):
        cols = st.columns(3, gap="medium")
        for col, item in zip(cols, objectives[start:start + 3]):
            number, title, description = item
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card" style="margin-bottom:12px;">
                        <div class="feature-icon">{number}</div>
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")

    # 3. Technology stack
    section_label("03 · TECHNOLOGY STACK")
    st.subheader("Tools and technologies")

    technology = pd.DataFrame([
        {
            "Technology": "Python",
            "Purpose": "Main programming language for data processing and application logic",
        },
        {
            "Technology": "TensorFlow / Keras",
            "Purpose": "Model loading, training, fine-tuning and prediction",
        },
        {
            "Technology": "EfficientNetB0",
            "Purpose": "Transfer-learning architecture for image classification",
        },
        {
            "Technology": "NumPy",
            "Purpose": "Numerical operations and image-array processing",
        },
        {
            "Technology": "Pandas",
            "Purpose": "Dataset metadata and evaluation tables",
        },
        {
            "Technology": "Matplotlib",
            "Purpose": "Confusion matrices and evaluation visualizations",
        },
        {
            "Technology": "Pillow",
            "Purpose": "Opening, converting and resizing uploaded images",
        },
        {
            "Technology": "Streamlit",
            "Purpose": "Interactive web application and result displays",
        },
        {
            "Technology": "GitHub + Streamlit Cloud",
            "Purpose": "Source-code hosting and website deployment",
        },
    ])

    st.dataframe(
        technology,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "Model note: Experiment 4 V2 is a candidate checkpoint. "
        "The final model should be selected after consistent evaluation "
        "on the same data split."
    )

    st.write("")

    # 4. Dataset
    section_label("04 · DATASET DESCRIPTION")
    st.subheader("HAM10000 dataset")

    d1, d2, d3 = st.columns(3, gap="medium")
    d1.metric("Total images", "10,015")
    d2.metric("Lesion categories", "7")
    d3.metric("Image type", "Dermoscopic")

    st.markdown("""
    <div class="panel">
        HAM10000 stands for <i>Human Against Machine with 10,000 training
        images</i>. It is a publicly available collection of dermoscopic
        images used in skin-lesion classification research.
        The associated metadata contains diagnostic class labels for the
        images. The classes are not equally represented, which makes
        class-wise evaluation important.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Dataset classes")

    class_data = pd.DataFrame([
        {
            "Class code": "akiec",
            "Description": "Actinic keratoses and intraepithelial carcinoma",
        },
        {
            "Class code": "bcc",
            "Description": "Basal cell carcinoma",
        },
        {
            "Class code": "bkl",
            "Description": "Benign keratosis-like lesions",
        },
        {
            "Class code": "df",
            "Description": "Dermatofibroma",
        },
        {
            "Class code": "mel",
            "Description": "Melanoma",
        },
        {
            "Class code": "nv",
            "Description": "Melanocytic nevi",
        },
        {
            "Class code": "vasc",
            "Description": "Vascular lesions",
        },
    ])

    st.dataframe(
        class_data,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "The classifier predicts one of these dataset categories. "
        "The predicted category is not a medical diagnosis."
    )

    st.write("")

    # 5. Methodology
    section_label("05 · PROJECT METHODOLOGY")
    st.subheader("How the project is developed")

    methodology = [
        (
            "01 · Dataset collection",
            "Use HAM10000 images and metadata to obtain the image files "
            "and corresponding class labels.",
        ),
        (
            "02 · Image preprocessing",
            "Match images with labels, convert images to RGB, and resize "
            "them to the input dimensions expected by the model.",
        ),
        (
            "03 · Dataset splitting",
            "Separate training, validation and test data. Grouping images "
            "by lesion when splitting helps reduce data leakage.",
        ),
        (
            "04 · Transfer learning",
            "Use EfficientNetB0 to reuse pretrained visual features for "
            "the skin-lesion classification task.",
        ),
        (
            "05 · Training and fine-tuning",
            "Train the classification head and conduct fine-tuning "
            "experiments to study model performance.",
        ),
        (
            "06 · Model evaluation",
            "Calculate appropriate classification metrics and inspect "
            "the confusion matrix and per-class performance.",
        ),
        (
            "07 · Web application",
            "Load the selected model in Streamlit and provide image upload, "
            "prediction scores and evaluation information.",
        ),
        (
            "08 · Deployment",
            "Host the application using Streamlit Community Cloud and "
            "maintain the code and required files in GitHub.",
        ),
    ]

    for title, description in methodology:
        with st.expander(title):
            st.write(description)

    st.write("")

    # 6. System workflow
    section_label("06 · SYSTEM WORKFLOW")
    st.subheader("From image upload to model output")

    workflow = [
        ("1", "Upload image", "JPG / PNG"),
        ("2", "Preprocess", "RGB · 224 × 224"),
        ("3", "Run model", "EfficientNetB0"),
        ("4", "Get scores", "Seven classes"),
        ("5", "Display results", "Category + scores"),
    ]

    workflow_cols = st.columns(5, gap="small")
    for col, (number, title, description) in zip(workflow_cols, workflow):
        with col:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div class="feature-icon" style="margin:0 auto 12px auto;">
                        {number}
                    </div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("""
    <div class="panel">
        <b>How prediction works:</b>
        The uploaded image is converted to RGB format and resized to
        224 × 224 pixels. The trained model processes the image and
        returns an output score for each of the seven classes. The app
        displays the highest-scoring category along with the other scores.
        These scores are model outputs, not necessarily calibrated
        probabilities of disease.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # 7. Evaluation
    section_label("07 · MODEL EVALUATION")
    st.subheader("How performance is measured")

    metric_cols = st.columns(2, gap="medium")

    with metric_cols[0]:
        st.markdown("""
        <div class="feature-card">
            <h3>Accuracy</h3>
            <p>The proportion of evaluated images classified correctly.</p>
            <hr>
            <h3>Precision</h3>
            <p>Among predictions for a class, the proportion that is correct.</p>
            <hr>
            <h3>Recall</h3>
            <p>Among actual examples of a class, the proportion identified correctly.</p>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[1]:
        st.markdown("""
        <div class="feature-card">
            <h3>Macro F1-score</h3>
            <p>The average F1-score across classes, giving each class equal weight.</p>
            <hr>
            <h3>Weighted F1-score</h3>
            <p>The class-wise F1-scores averaged according to class support.</p>
            <hr>
            <h3>Confusion matrix</h3>
            <p>A table showing correct predictions and the classes confused with one another.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="panel">
        <b>Interpreting results:</b>
        Overall accuracy alone may hide poor performance on categories
        with fewer examples. For this reason, macro F1-score, class-wise
        recall and the confusion matrix are important when comparing
        candidate models.
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Use the Model Evaluation page to view the metrics currently "
        "recorded in the project. Verify each figure against the matching "
        "model checkpoint and evaluation split before reporting final results."
    )

    st.write("")

    # 8. Application features
    section_label("08 · WEBSITE FEATURES")
    st.subheader("What users can explore")

    features = [
        ("Overview", "Project introduction, technology and dataset summary."),
        ("AI Prediction", "Upload a compatible image and inspect model output scores."),
        ("Model Evaluation", "Review the recorded metrics, classification report and confusion matrix."),
        ("About Project", "Explore methodology, workflow, technologies and limitations."),
    ]

    for title, description in features:
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:10px;">
                <b>{title}</b><br>
                <span style="color:#718297;">{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # 9. Limitations
    section_label("09 · LIMITATIONS & RESPONSIBLE USE")
    st.subheader("What the project cannot establish")

    limitations = [
        "The dataset may not represent all skin types, populations, imaging devices or real-world clinical settings.",
        "The number of images differs across classes and may affect performance.",
        "Different lesion categories can have similar visual features.",
        "The model can make incorrect predictions, including on potentially serious lesions.",
        "Performance on a research dataset does not establish clinical safety or effectiveness.",
        "The website has not been established as a clinically validated diagnostic system.",
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.warning(
        "DermaAI is for educational research only. Do not use its output "
        "to diagnose a condition, choose treatment, or decide whether to "
        "seek care. A qualified clinician should assess concerning or "
        "changing skin lesions."
    )

    st.write("")

    # 10. Future scope
    section_label("10 · FUTURE SCOPE")
    st.subheader("Possible improvements")

    future_work = [
        "Compare multiple architectures using the same evaluation split.",
        "Investigate methods for improving performance on underrepresented classes.",
        "Use consistent lesion-level splitting and document the evaluation protocol.",
        "Record model versions and experiment settings for reproducibility.",
        "Improve charts, accessibility and clarity of prediction displays.",
        "Consider external validation before any proposed clinical application.",
    ]

    for item in future_work:
        st.markdown(f"- {item}")

    st.write("")

    # 11. Team
    section_label("11 · PROJECT TEAM")
    team1, team2 = st.columns(2, gap="medium")

    with team1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">👨‍💻</div>
            <h3>Team leader</h3>
            <p>Shivam Kumar</p>
        </div>
        """, unsafe_allow_html=True)

    with team2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤝</div>
            <h3>Team members</h3>
            <p>Aryan Rawat, Aditya Chauhan and Zaid</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 12. Conclusion
    section_label("12 · CONCLUSION")
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">PROJECT SUMMARY</div>
        <h1 style="font-size:2rem !important;">Learning through responsible AI</h1>
        <p>
        DermaAI demonstrates an educational workflow that combines
        dataset preparation, transfer learning, model evaluation and
        Streamlit deployment. It provides practical experience with
        Python, TensorFlow, image processing and web development while
        highlighting the importance of careful evaluation and responsible
        interpretation of medical-image model outputs.
        </p>
        <div class="hero-pill">Research prototype · Not a medical device</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Project team: Shivam Kumar, Aryan Rawat, Aditya Chauhan and Zaid."
    )

        "Dataset performance does not establish clinical safety. "
        "Performance can vary across devices, populations, image "
        "types and clinical settings. This prototype is not intended "
        "to diagnose skin cancer or replace a qualified clinician."
    )
