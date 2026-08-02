from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Page configuration
# ============================================================
st.set_page_config(
    page_title="Low STF Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. Visual system
#    UI-only styling: model inputs and prediction logic are unchanged.
# ============================================================
st.markdown(
    """
    <style>
        :root {
            --stf-canvas: #f4f7fb;
            --stf-surface: #ffffff;
            --stf-navy: #102447;
            --stf-text: #213652;
            --stf-muted: #63738d;
            --stf-border: #d8e1ed;
            --stf-blue: #2574e8;
            --stf-teal: #0a8f91;
            --stf-teal-dark: #087477;
            --stf-soft-blue: #eef5ff;
            --stf-soft-teal: #eaf8f7;
            --stf-warning: #b25f16;
        }

        .stApp {
            background: var(--stf-canvas);
            color: var(--stf-text);
        }

        .block-container {
            max-width: 1504px;
            padding: 1.6rem 2.5rem 3rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            height: 0;
            min-height: 0;
            background: transparent;
        }

        div[data-testid="stDecoration"] {
            display: none;
        }

        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"],
        #MainMenu {
            display: none;
        }

        /* Quiet application header */
        .stf-header {
            margin: 0 0 1rem 0;
        }

        .stf-header h1 {
            margin: 0;
            color: var(--stf-navy);
            font-size: clamp(2rem, 3.2vw, 3rem);
            font-weight: 750;
            line-height: 1.12;
            letter-spacing: -0.035em;
        }

        .stf-header p {
            max-width: 780px;
            margin: 0.65rem 0 0;
            color: var(--stf-muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }

        /* Tabs */
        div[data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 1px solid var(--stf-border);
        }

        button[data-baseweb="tab"],
        div[data-testid="stTab"] {
            min-height: 52px;
            padding-right: 0.2rem;
            padding-left: 0.2rem;
            color: var(--stf-muted);
            font-size: 0.98rem;
            font-weight: 650;
        }

        button[data-baseweb="tab"][aria-selected="true"],
        div[data-testid="stTab"][aria-selected="true"] {
            color: var(--stf-blue) !important;
            border-bottom-color: var(--stf-blue) !important;
        }

        div[data-baseweb="tab-highlight"],
        div[data-testid="stTab"] .react-aria-SelectionIndicator {
            background-color: var(--stf-blue) !important;
        }

        /* Form surface */
        div[data-testid="stForm"] {
            margin-top: 1rem;
            padding: 1.2rem 1.4rem 1.1rem;
            border: 1px solid var(--stf-border);
            border-radius: 16px;
            background: var(--stf-surface);
            box-shadow: 0 10px 28px rgba(16, 36, 71, 0.055);
        }

        .stf-section-heading {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.1rem 0 0.75rem;
        }

        .stf-section-icon {
            display: inline-flex;
            flex: 0 0 auto;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            border: 1px solid #cfe0f8;
            border-radius: 10px;
            background: var(--stf-soft-blue);
            color: var(--stf-blue);
        }

        .stf-section-icon.teal {
            border-color: #cce9e7;
            background: var(--stf-soft-teal);
            color: var(--stf-teal);
        }

        .stf-section-icon svg {
            width: 21px;
            height: 21px;
            stroke: currentColor;
        }

        .stf-section-copy h2 {
            margin: 0.05rem 0 0;
            color: var(--stf-navy);
            font-size: 1.08rem;
            font-weight: 720;
            line-height: 1.3;
        }

        .stf-section-copy p {
            margin: 0.25rem 0 0;
            color: var(--stf-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        /* Form labels and controls */
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            color: var(--stf-text);
            font-size: 0.84rem;
            font-weight: 620;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInputContainer"],
        div[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"] {
            min-height: 44px;
            border-color: #cbd6e5;
            border-radius: 10px;
            background: #ffffff !important;
            box-shadow: none;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stNumberInputContainer"]:focus-within,
        div[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"]:focus-within {
            border-color: var(--stf-blue);
            box-shadow: 0 0 0 3px rgba(37, 116, 232, 0.11);
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] * {
            color: var(--stf-text);
            font-size: 0.94rem;
        }

        /* Primary action */
        div[data-testid="stFormSubmitButton"] {
            max-width: 660px;
            margin: 0.8rem auto 0;
        }

        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 50px;
            border: 1px solid var(--stf-teal);
            border-radius: 11px;
            background: var(--stf-teal) !important;
            color: #ffffff !important;
            font-size: 0.97rem;
            font-weight: 700;
            letter-spacing: 0.005em;
            box-shadow: 0 8px 18px rgba(10, 143, 145, 0.18);
            transition: transform 150ms ease, background 150ms ease,
                        box-shadow 150ms ease;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--stf-teal-dark);
            background: var(--stf-teal-dark) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(10, 143, 145, 0.24);
        }

        div[data-testid="stFormSubmitButton"] button:focus {
            background: var(--stf-teal) !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 4px rgba(10, 143, 145, 0.16);
        }

        /* Results */
        .stf-results-heading {
            margin-top: 1.25rem;
        }

        div[data-testid="stMetric"] {
            min-height: 110px;
            padding: 1rem 1.15rem;
            border: 1px solid var(--stf-border);
            border-radius: 14px;
            background: var(--stf-surface);
            box-shadow: 0 7px 20px rgba(16, 36, 71, 0.045);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--stf-muted);
            font-size: 0.9rem;
            font-weight: 620;
        }

        div[data-testid="stMetricValue"] {
            color: var(--stf-teal);
            font-size: 2.15rem;
            font-weight: 760;
            letter-spacing: -0.025em;
        }

        div[data-testid="stProgress"] > div > div > div {
            background-color: var(--stf-teal);
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        .stf-research-note {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-top: 1rem;
            padding: 0.85rem 1rem;
            border: 1px solid var(--stf-border);
            border-radius: 11px;
            background: #ffffff;
            color: var(--stf-muted);
            font-size: 0.82rem;
        }

        .stf-research-note svg {
            width: 18px;
            height: 18px;
            flex: 0 0 auto;
            stroke: var(--stf-blue);
        }

        /* Methodology */
        .stf-flow {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
            margin: 1rem 0 1.25rem;
        }

        .stf-flow-step {
            padding: 1rem;
            border: 1px solid var(--stf-border);
            border-radius: 12px;
            background: #ffffff;
        }

        .stf-flow-step strong {
            display: block;
            margin-bottom: 0.25rem;
            color: var(--stf-navy);
            font-size: 0.9rem;
        }

        .stf-flow-step span {
            color: var(--stf-muted);
            font-size: 0.8rem;
        }

        div[data-testid="stImage"] img {
            border: 1px solid var(--stf-border);
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(16, 36, 71, 0.05);
        }

        footer {
            visibility: hidden;
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 1.15rem 1rem 2.5rem;
            }

            .stf-header h1 {
                font-size: 2rem;
            }

            div[data-testid="stForm"] {
                padding: 1.1rem 1rem;
            }

            .stf-flow {
                grid-template-columns: 1fr;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                scroll-behavior: auto !important;
                transition: none !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Reusable code-native SVG icons and headings
# ============================================================
ICONS = {
    "waveform": """
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M3 12h3l2.1-6.5 3.6 13L14 9l1.7 3H21" />
        </svg>
    """,
    "patient": """
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="7" r="3.2" />
            <path d="M5 20c.5-4 3-6.2 7-6.2s6.5 2.2 7 6.2" />
        </svg>
    """,
    "target": """
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="11" cy="13" r="7" />
            <circle cx="11" cy="13" r="3" />
            <path d="m14 10 6-6m-4 0h4v4" />
        </svg>
    """,
    "method": """
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M6 3h9l3 3v15H6z" />
            <path d="M15 3v4h4M9 11h6M9 15h3" />
            <circle cx="16.5" cy="16.5" r="2.5" />
        </svg>
    """,
    "info": """
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 11v6M12 7.5h.01" />
        </svg>
    """,
}


def section_heading(
    title: str,
    description: str,
    icon: str,
    *,
    teal: bool = False,
    results: bool = False,
) -> None:
    """Render a consistent section heading with an inline SVG icon."""
    color_class = " teal" if teal else ""
    results_class = " stf-results-heading" if results else ""
    icon_svg = "".join(line.strip() for line in ICONS[icon].splitlines())
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f'<div class="stf-section-heading{results_class}">'
        f'<span class="stf-section-icon{color_class}">{icon_svg}</span>'
        f'<div class="stf-section-copy"><h2>{title}</h2>'
        f'{description_html}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 4. Model and asset loading
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent


def find_model_paths() -> Tuple[Path, Path]:
    """Find the existing model files without changing their expected names."""
    candidate_directories = (
        SCRIPT_DIR / "STF-Deploy",
        SCRIPT_DIR,
        Path.cwd() / "STF-Deploy",
        Path.cwd(),
    )

    for directory in candidate_directories:
        rf_path = directory / "rf_model.pkl"
        lr_path = directory / "combined_model.pkl"
        if rf_path.is_file() and lr_path.is_file():
            return rf_path, lr_path

    expected = SCRIPT_DIR / "STF-Deploy"
    raise FileNotFoundError(
        "Could not find rf_model.pkl and combined_model.pkl together. "
        f"Place both files in: {expected}"
    )


if hasattr(st, "cache_resource"):
    cache_models = st.cache_resource(show_spinner=False)
else:
    # Compatibility with older Streamlit releases.
    cache_models = st.cache(allow_output_mutation=True, show_spinner=False)


@cache_models
def load_models(rf_path: str, lr_path: str):
    """Load the original two-stage prediction models once per app session."""
    rf_model = joblib.load(rf_path)
    lr_model = joblib.load(lr_path)
    return rf_model, lr_model


try:
    rf_model_path, lr_model_path = find_model_paths()
    rf_model, lr_model = load_models(str(rf_model_path), str(lr_model_path))
    model_error = None
except Exception as exc:
    rf_model = None
    lr_model = None
    model_error = str(exc)


def find_figure() -> Optional[Path]:
    for path in (SCRIPT_DIR / "Figure.png", Path.cwd() / "Figure.png"):
        if path.is_file():
            return path
    return None


# ============================================================
# 5. Application header and navigation
# ============================================================
st.markdown(
    """
    <header class="stf-header">
        <h1>Multimodal Prediction System for Low STF</h1>
        <p>
            Radiomics and clinical signatures combined in a two-stage
            prediction workflow.
        </p>
    </header>
    """,
    unsafe_allow_html=True,
)

prediction_tab, methodology_tab = st.tabs(["Prediction", "Methodology"])


# ============================================================
# 6. Prediction interface
# ============================================================
with prediction_tab:
    if model_error:
        st.error(
            "The prediction models could not be loaded. "
            f"Check the model files and paths. Details: {model_error}"
        )

    with st.form("low_stf_prediction_form"):
        radiomics_panel, clinical_panel = st.columns([2, 1], gap="large")

        with radiomics_panel:
            section_heading(
                "Radiomics features",
                "",
                "waveform",
            )

            radiomics_col_1, radiomics_col_2 = st.columns(2, gap="medium")

            with radiomics_col_1:
                feat_1 = st.number_input(
                    "exp_ngtdm_Busyness", value=0.0, format="%.6f"
                )
                feat_2 = st.number_input(
                    "log_glcm_ClusterShade", value=0.0, format="%.6f"
                )
                feat_3 = st.number_input(
                    "log_glcm_DifferenceAverage", value=0.0, format="%.6f"
                )
                feat_4 = st.number_input(
                    "square_glcm_Imc1", value=0.0, format="%.6f"
                )

            with radiomics_col_2:
                feat_5 = st.number_input(
                    "sqrt_firstorder_Skewness", value=0.0, format="%.6f"
                )
                feat_6 = st.number_input(
                    "sqrt_glrlm_ShortRunLowGrayLevel",
                    value=0.0,
                    format="%.6f",
                )
                feat_7 = st.number_input(
                    "wavelet_LHL_firstorder_Skewness",
                    value=0.0,
                    format="%.6f",
                )
                feat_8 = st.number_input(
                    "wavelet_LLL_ngtdm_Strength", value=0.0, format="%.6f"
                )

        with clinical_panel:
            section_heading(
                "Clinical signatures",
                "",
                "patient",
                teal=True,
            )

            sex_input = st.selectbox("Sex", options=["Male", "Female"])
            bmi_input = st.selectbox(
                "Body Mass Index (BMI)", options=["≤ 25 kg/m²", "> 25 kg/m²"]
            )
            nlr_input = st.selectbox("NLR", options=["≤ 3.0", "> 3.0"])
            age_input = st.selectbox(
                "Age", options=["≤ 65 years", "> 65 years"]
            )

        run_prediction = st.form_submit_button(
            "Run Low STF Prediction", use_container_width=True
        )

    # The following preprocessing and two-stage prediction logic matches
    # the original application. Only the visible terminology has changed.
    if run_prediction:
        if rf_model is None or lr_model is None:
            st.error("Prediction is unavailable until both model files are loaded.")
        else:
            with st.spinner("Processing multimodal data..."):
                sex_val = 1 if sex_input == "Male" else 2
                bmi_val = 0 if bmi_input == "≤ 25 kg/m²" else 1
                nlr_val = 0 if nlr_input == "≤ 3.0" else 1
                age_val = 0 if age_input == "≤ 65 years" else 1

                radiomics_data = np.array(
                    [[
                        feat_1,
                        feat_2,
                        feat_3,
                        feat_4,
                        feat_5,
                        feat_6,
                        feat_7,
                        feat_8,
                    ]]
                )
                rad_score_prob = rf_model.predict_proba(radiomics_data)[0][1]

                lr_input_df = pd.DataFrame(
                    {
                        "Sex": [sex_val],
                        "BMI": [bmi_val],
                        "NLR": [nlr_val],
                        "Age": [age_val],
                        "Radscore": [rad_score_prob],
                    }
                )
                final_prob = lr_model.predict_proba(lr_input_df)[0][1] * 100

                st.session_state["low_stf_prediction"] = {
                    "radiomics_probability": float(rad_score_prob * 100),
                    "final_probability": float(final_prob),
                }

    result = st.session_state.get("low_stf_prediction")
    if result is not None:
        rad_probability = result["radiomics_probability"]
        low_stf_probability = result["final_probability"]

        section_heading(
            "Prediction results",
            "",
            "target",
            results=True,
        )

        result_col_1, result_col_2 = st.columns(2, gap="medium")
        with result_col_1:
            st.metric(
                label="Radiomics probability",
                value=f"{rad_probability:.2f}%",
            )
            st.caption(
                "Derived from the eight non-linear quantitative imaging features."
            )

        with result_col_2:
            st.metric(
                label="Final Low STF probability",
                value=f"{low_stf_probability:.2f}%",
            )
            st.caption(
                "Combined estimate based on radiomics and clinical signatures."
            )

        st.progress(int(round(low_stf_probability)))

        if low_stf_probability > 50:
            st.warning(
                "High probability of Low STF detected "
                f"({low_stf_probability:.2f}%)."
            )
        else:
            st.success(
                "Low probability of Low STF "
                f"({low_stf_probability:.2f}%)."
            )

        info_icon = "".join(line.strip() for line in ICONS["info"].splitlines())
        st.markdown(
            '<div class="stf-research-note">'
            f'{info_icon}'
            '<span>For research use only. Not for clinical decision-making.</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 7. Methodology interface
# ============================================================
with methodology_tab:
    section_heading(
        "Methodological framework",
        "Image processing, radiomics quantification, and clinical integration",
        "method",
    )

    st.markdown(
        """
        <div class="stf-flow">
            <div class="stf-flow-step">
                <strong>Stage I · Imaging</strong>
                <span>Image super-resolution and radiomics feature extraction</span>
            </div>
            <div class="stf-flow-step">
                <strong>Stage II · Integration</strong>
                <span>Radiomics probability combined with clinical signatures</span>
            </div>
            <div class="stf-flow-step">
                <strong>Output · Low STF</strong>
                <span>Final probability generated by the combined model</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    figure_path = find_figure()
    if figure_path is None:
        st.info(
            "Add Figure.png beside this script to display the complete "
            "methodological framework."
        )
    else:
        try:
            st.image(
                str(figure_path),
                caption="Methodological framework for Low STF prediction",
                use_container_width=True,
            )
        except TypeError:
            # Compatibility with older Streamlit releases.
            st.image(
                str(figure_path),
                caption="Methodological framework for Low STF prediction",
                use_column_width=True,
            )
