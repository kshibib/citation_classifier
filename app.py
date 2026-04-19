"""Streamlit app for online legal citation classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st
from st_keyup import st_keyup

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

FEDERAL_LABELS = {
    "federal_app_case",
    "federal_constitution",
    "federal_docket",
    "federal_executive_order",
    "federal_regulation",
    "federal_rule",
    "federal_statute",
    "federal_trial_case",
}

STATE_LABELS = {
    "state_app_case",
    "state_constitution",
    "state_executive_order",
    "state_regulation",
    "state_rule",
    "state_statute",
    "state_trial_case",
}


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


def render_prediction(prediction: str) -> None:
    """Render the prediction using label-specific colors."""
    if prediction in FEDERAL_LABELS:
        color = "#1f5aa6"
        background = "transparent"
    elif prediction in STATE_LABELS:
        color = "#c62828"
        background = "transparent"
    elif prediction == "case":
        color = "#7b1fa2"
        background = "transparent"
    elif prediction == "unknown":
        color = "#ffffff"
        background = "#333333"
    else:
        color = "#14243d"
        background = "transparent"

    st.markdown(
        f"""
        <div style="margin-top:0.5rem;">
          <span style="
            color:{color};
            background:{background};
            font-weight:700;
            font-size:1.1rem;
            padding:0.2rem 0.45rem;
            border-radius:0.35rem;
            display:inline-block;
          ">{prediction}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Shibib's ML Citation Classifier",
        page_icon="L",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        html, body, .stApp, .stApp *, [class*="css"], [class*="st-"], [data-testid] {
            font-family: "Times New Roman", Times, serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Shibib's ML Citation Classifier")
    st.caption(
        "Paste a single legal citation and classify it below."
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

    citation = st_keyup(
        "Citation",
        value="",
        placeholder="Example: 410 U.S. 113 (1973)",
        key="citation_input",
    )

    cleaned = citation.strip()
    if cleaned:
        prediction = model.predict([cleaned])[0]
        st.subheader("Prediction")
        render_prediction(prediction)
    else:
        st.info("Enter a citation to see a prediction.")


if __name__ == "__main__":
    main()
