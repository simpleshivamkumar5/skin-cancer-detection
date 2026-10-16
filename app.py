    
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
    - EfficientNetB0 (current candidate: Experiment 4 V2)
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
