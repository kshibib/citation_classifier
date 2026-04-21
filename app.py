"""Streamlit app for online legal citation classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st
from st_keyup import st_keyup

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "KS_Icon_Transparent.png"

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
        color = "#5fb8ff"
    elif prediction in STATE_LABELS:
        color = "#ff6b6b"
    elif prediction == "case":
        color = "#7b1fa2"
    elif prediction == "unknown":
        color = "#c7ccd6"
    else:
        color = "#3f8635"

    st.markdown(
        f"""
        <div class="prediction-value">
          <span class="prediction-text" style="color:{color};">{prediction}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Shibib’s ML Citation Classifier",
        page_icon="L",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        :root {
            --size-base: 1rem;
            --size-title: 2.25rem;
            --size-prediction-label: 1.15rem;
            --size-prediction-value: 1.35rem;
            --weight-bold: 700;
        }
        html {
            font-size: 16px;
        }
        html, body, .stApp, .stApp *, [class*="css"], [class*="st-"], [data-testid],
        [data-baseweb], input, textarea, select, option, button, label, p, span, div {
            font-size: var(--size-base);
        }
        .app-title {
            margin: 0 0 0.25rem 0;
            font-size: var(--size-title) !important;
            font-weight: var(--weight-bold);
            line-height: 1.15;
        }
        .prediction-label {
            margin-top: 1rem;
            padding-left: 0.45rem;
            font-size: var(--size-prediction-label) !important;
            font-weight: var(--weight-bold);
            line-height: 1.2;
        }
        .prediction-value {
            margin-top: 0.5rem;
        }
        .prediction-text {
            font-size: var(--size-prediction-value) !important;
            font-weight: var(--weight-bold);
            padding: 0.2rem 0.45rem;
            border-radius: 0.35rem;
            display: inline-block;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stWidgetLabel"] p,
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] span,
        div[data-baseweb="input"] > div,
        input[type="text"],
        input[type="text"]::placeholder,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stAlertContent"],
        [data-testid="stSpinner"] * {
            font-size: var(--size-base) !important;
        }
        [data-testid="stWidgetLabel"] p {
            font-weight: var(--weight-bold);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input[type="text"] {
            min-height: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_col = st.columns([1, 1.4, 1])[1]
    with logo_col:
        st.image(str(LOGO_PATH), width=320)

    st.markdown(
        '<h1 class="app-title">ML Citation Classifier</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Paste a single legal citation and classify it below.")

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
        disabled=(len(available_models) == 1),
    )

    with st.spinner("Loading model..."):
        model, _label_encoder = load_model_bundle(selected_model)

    citation = st_keyup(
        "",
        value="",
        placeholder="",
        key="citation_input",
    )

    st.markdown(
        '<div class="prediction-label">Prediction:</div>',
        unsafe_allow_html=True,
    )

    cleaned = citation.strip()
    if cleaned:
        prediction = model.predict([cleaned])[0]
        render_prediction(prediction)


if __name__ == "__main__":
    main()
