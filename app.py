"""Streamlit app for online legal citation classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_DIRS = [
    ROOT / "deployed_models" / "linear_svm",
    ROOT / "outputs" / "linear_svm_run_augmented_rerun",
    ROOT / "outputs" / "linear_svm_run_augmented",
    ROOT / "outputs" / "linear_svm_run",
    ROOT / "outputs" / "logistic_run_augmented_rerun",
    ROOT / "outputs" / "logistic_run_augmented",
    ROOT / "outputs" / "logistic_run",
]


def discover_model_dirs() -> list[Path]:
    """Return model directories that contain both required artifacts."""
    found: list[Path] = []
    for directory in DEFAULT_MODEL_DIRS:
        if (directory / "model.joblib").exists() and (
            directory / "label_encoder.joblib"
        ).exists():
            found.append(directory)
    return found


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_dir: str) -> tuple[object, object]:
    """Load a saved model and label encoder."""
    directory = Path(model_dir)
    model = joblib.load(directory / "model.joblib")
    label_encoder = joblib.load(directory / "label_encoder.joblib")
    return model, label_encoder


def main() -> None:
    st.set_page_config(
        page_title="Legal Citation Classifier",
        page_icon="L",
        layout="centered",
    )

    st.title("Legal Citation Classifier")
    st.caption(
        "Paste a single legal citation and classify it with the saved baseline model."
    )

    available_models = discover_model_dirs()
    if not available_models:
        st.error(
            "No saved model artifacts were found. Add a deployable model under "
            "`deployed_models/linear_svm/` or commit a saved run directory."
        )
        st.stop()

    model_labels = {
        str(path): path.relative_to(ROOT).as_posix() for path in available_models
    }
    selected_model = st.selectbox(
        "Model",
        options=list(model_labels.keys()),
        format_func=lambda path: model_labels[path],
    )

    with st.spinner("Loading model..."):
        model, label_encoder = load_model_bundle(selected_model)

    citation = st.text_area(
        "Citation",
        height=160,
        placeholder="Example: 410 U.S. 113 (1973)",
    )

    classify = st.button("Classify", type="primary")

    if classify:
        cleaned = citation.strip()
        if not cleaned:
            st.warning("Enter a citation first.")
        else:
            prediction = model.predict([cleaned])[0]
            st.subheader("Prediction")
            st.code(prediction, language="text")

            with st.expander("Available labels"):
                st.write(", ".join(label_encoder.classes_.tolist()))

    st.divider()
    st.markdown(
        """
        **Deploy notes**

        This app expects two files in the selected model directory:
        `model.joblib` and `label_encoder.joblib`.
        For GitHub deployment, the easiest option is to commit one model directory
        under `deployed_models/linear_svm/`.
        """
    )


if __name__ == "__main__":
    main()
