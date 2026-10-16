
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
