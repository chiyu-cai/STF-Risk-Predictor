from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import streamlit as st


# ============================================================
# 1. Page configuration
# ============================================================
st.set_page_config(
    page_title="Low STF Radiomics Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. Visual system
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
        }

        .stApp {
            background: var(--stf-canvas);
            color: var(--stf-text);
        }

        .block-container {
            max-width: 1180px;
            padding: 1.8rem 2.5rem 3rem;
        }

        header[data-testid="stHeader"] {
            height: 0;
            min-height: 0;
            background: transparent;
        }

        div[data-testid="stDecoration"],
        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"],
        #MainMenu {
            display: none;
        }

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
            max-width: 820px;
            margin: 0.65rem 0 0;
            color: var(--stf-muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }

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

        div[data-testid="stForm"] {
            margin-top: 1rem;
            padding: 1.35rem 1.5rem 1.2rem;
            border: 1px solid var(--stf-border);
            border-radius: 16px;
            background: var(--stf-surface);
            box-shadow: 0 10px 28px rgba(16, 36, 71, 0.055);
        }

        .stf-section-heading {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.1rem 0 0.8rem;
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

        div[data-testid="stNumberInput"] label {
            color: var(--stf-text);
            font-size: 0.84rem;
            font-weight: 620;
        }

        div[data-baseweb="input"] > div,
        div[data-testid="stNumberInputContainer"] {
            min-height: 44px;
            border-color: #cbd6e5;
            border-radius: 10px;
            background: #ffffff !important;
            box-shadow: none;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-testid="stNumberInputContainer"]:focus-within {
            border-color: var(--stf-blue);
            box-shadow: 0 0 0 3px rgba(37, 116, 232, 0.11);
        }

        div[data-baseweb="input"] input {
            color: var(--stf-text);
            font-size: 0.94rem;
        }

        div[data-testid="stFormSubmitButton"] {
            max-width: 560px;
            margin: 1rem auto 0;
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
            box-shadow: 0 8px 18px rgba(10, 143, 145, 0.18);
            transition: transform 150ms ease, background 150ms ease,
                        box-shadow 150ms ease;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--stf-teal-dark);
            background: var(--stf-teal-dark) !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(10, 143, 145, 0.24);
        }

        .stf-results-heading {
            margin-top: 1.25rem;
        }

        .stf-result-wrap {
            max-width: 680px;
            margin: 0 auto;
        }

        div[data-testid="stMetric"] {
            min-height: 120px;
            padding: 1.05rem 1.2rem;
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
            font-size: 2.2rem;
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
# 3. Reusable SVG icons and headings
# ============================================================
ICONS = {
    "waveform": """
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M3 12h3l2.1-6.5 3.6 13L14 9l1.7 3H21" />
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


def find_model_path() -> Path:
    """Find the radiomics-only random-forest model."""
    candidate_directories = (
        SCRIPT_DIR / "STF-Deploy",
        SCRIPT_DIR,
        Path.cwd() / "STF-Deploy",
        Path.cwd(),
    )

    for directory in candidate_directories:
        rf_path = directory / "rf_model.pkl"
        if rf_path.is_file():
            return rf_path

    expected = SCRIPT_DIR / "STF-Deploy" / "rf_model.pkl"
    raise FileNotFoundError(
        f"Could not find rf_model.pkl. Expected location: {expected}"
    )


if hasattr(st, "cache_resource"):
    cache_model = st.cache_resource(show_spinner=False)
else:
    cache_model = st.cache(allow_output_mutation=True, show_spinner=False)


@cache_model
def load_model(rf_path: str):
    """Load the radiomics model once per app session."""
    return joblib.load(rf_path)


try:
    rf_model_path = find_model_path()
    rf_model = load_model(str(rf_model_path))
    model_error = None
except Exception as exc:
    rf_model = None
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
        <h1>Radiomics-based Prediction System for Low STF</h1>
        <p>
            A proof-of-concept tool for individualized estimation of low-STF
            probability using eight quantitative CT radiomics features.
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
            "The radiomics model could not be loaded. "
            f"Check rf_model.pkl and its path. Details: {model_error}"
        )

    with st.form("low_stf_prediction_form"):
        section_heading(
            "Radiomics features",
            "Enter the eight radiomics features used by the trained random-forest model.",
            "waveform",
        )

        radiomics_col_1, radiomics_col_2 = st.columns(2, gap="large")

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

        run_prediction = st.form_submit_button(
            "Estimate Low STF Probability", use_container_width=True
        )

    if run_prediction:
        if rf_model is None:
            st.error("Prediction is unavailable until rf_model.pkl is loaded.")
        else:
            with st.spinner("Processing radiomics features..."):
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

                # The original app treated class 1 as low-STF status.
                # Keep this behavior unless the model's class definition differs.
                low_stf_prob = float(rf_model.predict_proba(radiomics_data)[0][1])

                st.session_state["low_stf_prediction"] = {
                    "probability": low_stf_prob * 100
                }

    result = st.session_state.get("low_stf_prediction")
    if result is not None:
        low_stf_probability = result["probability"]

        section_heading(
            "Prediction result",
            "",
            "target",
            teal=True,
            results=True,
        )

        st.markdown('<div class="stf-result-wrap">', unsafe_allow_html=True)
        st.metric(
            label="Estimated probability of low STF",
            value=f"{low_stf_probability:.2f}%",
        )
        st.caption(
            "Probability generated directly by the radiomics-only random-forest model."
        )
        st.progress(int(round(low_stf_probability)))

        # A 50% threshold is retained from the previous interface.
        # Replace this with the prespecified model cutoff if your manuscript uses another value.
        if low_stf_probability > 50:
            st.warning(
                "The model classifies this case as higher probability of low STF "
                f"({low_stf_probability:.2f}%)."
            )
        else:
            st.success(
                "The model classifies this case as lower probability of low STF "
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
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 7. Methodology interface
# ============================================================
with methodology_tab:
    section_heading(
        "Methodological framework",
        "Routine CT radiomics extraction and radiomics-only low-STF prediction",
        "method",
    )

    st.markdown(
        """
        <div class="stf-flow">
            <div class="stf-flow-step">
                <strong>Stage I · Radiomics</strong>
                <span>Quantitative features extracted from predefined thymic CT slices</span>
            </div>
            <div class="stf-flow-step">
                <strong>Stage II · Prediction</strong>
                <span>Eight selected features entered into the trained random-forest model</span>
            </div>
            <div class="stf-flow-step">
                <strong>Output · Low STF</strong>
                <span>Individualized probability of low-STF status</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    figure_path = find_figure()
    if figure_path is None:
        st.info(
            "Optional: add an updated radiomics-only Figure.png beside this script "
            "to display the methodological framework."
        )
    else:
        try:
            st.image(
                str(figure_path),
                caption="Radiomics-based framework for low-STF prediction",
                use_container_width=True,
            )
        except TypeError:
            st.image(
                str(figure_path),
                caption="Radiomics-based framework for low-STF prediction",
                use_column_width=True,
            )
