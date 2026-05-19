import streamlit as st
st.set_page_config(page_title='AI-Driven Real Estate Valuation System', layout='wide')

import html
import subprocess
import sys
from pathlib import Path
import time


def _assert_supported_python() -> None:
    """Fail fast with a clear message on unsupported Python versions."""
    version = sys.version_info[:2]
    if version < (3, 11) or version >= (3, 13):
        st.error(
            f"Unsupported Python {sys.version.split()[0]}. "
            "Use Python 3.11 or 3.12 for this project."
        )
        st.stop()


_assert_supported_python()

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Ensure the project root is on sys.path for stable sibling-package imports.
BASE_DIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BASE_DIR)
PROJECT_ROOT = Path(BASE_DIR)
DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "saved_model.pkl"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.pkl"
FEATURES_PATH = PROJECT_ROOT / "model" / "features.pkl"
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "model" / "train_model.py"


# Auto-train the model if artifacts are missing on first startup.
if not DATA_PATH.exists():
    st.error("Add house_prices.csv to data/")
    st.stop()

REQUIRED_MODEL_ARTIFACTS = [MODEL_PATH, METRICS_PATH, FEATURES_PATH]
if not all(path.exists() for path in REQUIRED_MODEL_ARTIFACTS):
    try:
        subprocess.run([sys.executable, str(TRAIN_SCRIPT_PATH)], check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()


# Import helper modules only after model availability is guaranteed.
from utils.explainer import get_shap_values
from utils.openrouter_api import explain_prediction


def safe_load_joblib(path: Path, name: str):
    """Load a joblib artifact safely and stop the app with a clear error if unavailable."""
    # Attempt to load the requested artifact with explicit error handling.
    try:
        return load_joblib_artifact(str(path))
    except Exception as exc:
        st.error(f"Failed to load {name} from {path}: {exc}")
        st.stop()


@st.cache_resource
def load_joblib_artifact(path: str):
    """Cache joblib artifact loading across reruns."""
    return joblib.load(path)


def safe_load_data(path: Path) -> pd.DataFrame:
    """Load dataset safely and provide a readable Streamlit error on failure."""
    # Attempt to load source data for slider ranges and fallback values.
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Failed to load dataset at {path}: {exc}")
        st.stop()


def apply_monochrome_theme(has_prediction: bool = False) -> None:
    """Inject a dark high-contrast theme and micro-interactions."""
    predict_pulse = "none" if has_prediction else "predictPulse 1.8s ease-in-out infinite"
    predict_shadow = (
        "0 0 0 1px var(--accent), 0 0 22px rgba(99,102,241,0.45)"
        if not has_prediction
        else "0 0 0 1px rgba(99,102,241,0.22)"
    )

    st.markdown(
        f"""
        <style>
        :root {{
            --bg-page:        #0F1117;
            --bg-card:        #151822;
            --bg-card-hover:  #1B1F2C;
            --bg-input:       #13151E;
            --bg-tag:         #1A1E2C;
            --border:         #2A2F3F;
            --border-strong:  #3C4258;
            --accent:         #7F77DD;
            --accent-hover:   #6B63C8;
            --accent-muted:   #16183A;
            --accent-text:    #B6B0F2;
            --text-primary:   #E2E2E8;
            --text-secondary: #9A9AB0;
            --text-muted:     #4A4A5A;
            --success:        #22C55E;
        }}

        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

        * {{
            font-family: 'Manrope', sans-serif;
        }}

        .stApp, .main, body, .block-container {{
            background: var(--bg-page) !important;
        }}

        .block-container {{
            padding: 2rem 3rem;
            max-width: 1380px;
        }}

        #MainMenu, header, footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stFooter"] {{
            visibility: hidden !important;
            height: 0 !important;
            position: fixed !important;
        }}

        p, li, span, label, .stMarkdown, .stCaption {{
            color: var(--text-secondary) !important;
        }}

        h1, h2, h3, .prediction-value-card .v, [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-family: 'DM Sans', sans-serif !important;
        }}

        .muted-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            font-family: 'Manrope', sans-serif !important;
            text-transform: uppercase;
            color: var(--text-muted) !important;
            margin: 0 0 0.45rem 0;
        }}

        .card-wrap {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            margin-bottom: 1rem;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }}

        .card-wrap:hover {{
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.65);
            box-shadow: 0 14px 26px rgba(0, 0, 0, 0.22);
        }}

        .st-key-card_property_summary,
        .st-key-card_feature_impact,
        .st-key-card_prediction_output,
        .st-key-card_comparable_homes,
        .st-key-card_ask_ai {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            margin-bottom: 1rem;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }}

        .st-key-card_property_summary:hover,
        .st-key-card_feature_impact:hover,
        .st-key-card_prediction_output:hover,
        .st-key-card_comparable_homes:hover,
        .st-key-card_ask_ai:hover {{
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.65);
            box-shadow: 0 14px 26px rgba(0, 0, 0, 0.22);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #161820 0%, #141722 100%) !important;
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebarCollapseButton"] {{
            display: none !important;
        }}

        .sidebar-top-title {{
            font-size: 42px;
            font-weight: 800;
            color: var(--text-primary) !important;
            margin: 0.1rem 0 0.35rem;
            letter-spacing: -0.04em;
            line-height: 1.02;
            font-family: 'DM Sans', sans-serif !important;
        }}

        .brand-tag {{
            font-size: 12px;
            color: var(--text-secondary) !important;
            margin: 0.1rem 0 0;
        }}

        .stepper {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.6rem 0 1rem;
            flex-wrap: wrap;
        }}

        .step {{
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            font-size: 12px;
            padding: 0.18rem 0.42rem;
            border-radius: 999px;
            color: var(--text-muted) !important;
            border: 1px solid transparent;
        }}

        .step .icon {{
            font-size: 11px;
            line-height: 1;
        }}

        .step.completed {{
            color: #86EFAC !important;
            border-color: rgba(34,197,94,0.35);
            background: rgba(34,197,94,0.08);
        }}

        .step.active {{
            color: var(--accent-text) !important;
            border-color: rgba(99,102,241,0.45);
            background: rgba(99,102,241,0.13);
            font-weight: 600;
        }}

        .step-sep {{
            color: var(--text-muted) !important;
            font-size: 12px;
        }}

        .st-key-preset_chip_group .stButton > button {{
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            background: var(--bg-tag) !important;
            color: var(--text-secondary) !important;
            font-size: 12px !important;
            padding: 0.45rem 0.65rem !important;
            min-height: 2.5rem !important;
            text-align: left !important;
        }}

        .st-key-preset_chip_group .stButton > button[kind="primary"] {{
            background: var(--accent-muted) !important;
            border-color: var(--accent) !important;
            color: var(--accent-text) !important;
            box-shadow: 0 0 0 1px rgba(99,102,241,0.22);
        }}

        .st-key-preset_chip_group .stButton > button p {{
            margin: 0 !important;
            font-size: 12px !important;
            color: inherit !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }}

        .slider-delta-chip {{
            margin: 0 0 0.5rem;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 11px;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            border: 1px solid transparent;
        }}

        .slider-delta-chip.pos {{
            color: #5DCAA5;
            border-color: rgba(29,158,117,0.35);
            background: rgba(29,158,117,0.12);
        }}

        .slider-delta-chip.neg {{
            color: #F09595;
            border-color: rgba(226,75,74,0.35);
            background: rgba(226,75,74,0.12);
        }}

        .st-key-predict_btn_wrap .stButton > button {{
            width: 100%;
            background: var(--accent) !important;
            color: #FFFFFF !important;
            border-radius: 9px !important;
            font-weight: 700 !important;
            padding: 0.64rem !important;
            border: none !important;
            box-shadow: {predict_shadow};
            animation: {predict_pulse};
            transition: transform 140ms ease, background 120ms ease, box-shadow 160ms ease;
        }}

        .st-key-predict_btn_wrap .stButton > button:hover {{
            background: var(--accent-hover) !important;
            transform: translateY(-1px);
        }}

        .prediction-ready-badge {{
            margin: 0.35rem 0 0;
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,0.35);
            background: rgba(34,197,94,0.1);
            color: #86EFAC;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}

        .prediction-ready-badge .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 0 rgba(34,197,94,0.65);
            animation: pulseDot 1.6s ease-out infinite;
        }}

        .stTextInput > div > div > input {{
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            font-size: 14px !important;
            padding: 0.5rem 1rem !important;
            background: var(--bg-input) !important;
            color: var(--text-primary) !important;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }}

        .stTextInput > div > div > input::placeholder {{
            color: var(--text-muted) !important;
        }}

        [data-testid="stMetric"],
        [data-testid="metric-container"] {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.7rem 0.8rem;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--text-muted) !important;
            font-size: 13px !important;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 24px !important;
            font-weight: 600 !important;
            font-variant-numeric: tabular-nums;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.15 !important;
            word-break: keep-all !important;
        }}

        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] > div > p {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-break: keep-all !important;
            overflow-wrap: anywhere !important;
        }}

        .slider-value-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.1rem 0 0.22rem;
            font-size: 12px;
            color: var(--text-secondary);
        }}

        .slider-value-row .name {{
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .slider-value-row .value {{
            color: var(--accent-text);
            font-weight: 600;
        }}

        /* Hide Streamlit floating thumb value and keep custom value row stable */
        [data-testid="stThumbValue"] {{
            display: none !important;
        }}

        .stSlider > div > div {{
            background: transparent !important;
        }}

        .stSlider [data-baseweb="slider"] > div > div {{
            background: var(--bg-input) !important;
            border-radius: 999px !important;
        }}

        .stSlider [data-baseweb="slider"]:hover {{
            box-shadow: 0 0 0 3px rgba(99,102,241,0.16);
            border-radius: 999px;
        }}

        .stSlider [data-baseweb="slider"] [role="slider"] {{
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 4px rgba(99,102,241,0.2) !important;
        }}

        .stSlider label,
        .stSlider p {{
            color: var(--text-secondary) !important;
        }}

        .chat-bubble {{
            background: var(--bg-input);
            border-radius: 8px;
            padding: 1rem;
            font-size: 14px;
            font-family: 'Manrope', sans-serif !important;
            line-height: 1.6;
            color: var(--text-primary);
            border: 1px solid var(--border);
        }}

        [data-testid="stAlert"] {{
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }}

        [data-testid="stAlert"] * {{
            color: var(--text-secondary) !important;
            font-size: 13px !important;
        }}

        .prediction-value-card {{
            background: var(--accent-muted);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--accent);
            box-shadow: 0 8px 18px rgba(30, 27, 75, 0.28);
        }}

        .prediction-value-card .k {{
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: var(--accent-text);
            text-transform: uppercase;
            margin: 0 0 4px;
        }}

        .prediction-value-card .v {{
            font-size: 36px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
            margin: 0 0 4px;
        }}

        .prediction-value-card .r {{
            font-size: 12px;
            color: var(--text-secondary);
            margin: 0;
        }}

        .confidence-wrap {{
            margin-top: 0.75rem;
        }}

        .confidence-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }}

        .confidence-track {{
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: #121522;
            border: 1px solid var(--border);
            overflow: hidden;
        }}

        .confidence-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #4F46E5 0%, #6366F1 45%, #A78BFA 100%);
            background-size: 200% 100%;
            animation: shimmer 3.2s linear infinite;
        }}

        .compare-table-wrap {{
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            background: var(--bg-input);
        }}

        .compare-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .compare-table th {{
            text-align: left;
            padding: 0.6rem 0.7rem;
            color: var(--accent-text);
            background: rgba(30, 34, 53, 0.8);
            border-bottom: 1px solid var(--border);
        }}

        .compare-table td {{
            padding: 0.55rem 0.7rem;
            color: var(--text-primary);
            border-bottom: 1px solid rgba(42,45,62,0.7);
        }}

        .compare-table tbody tr:nth-child(odd) {{
            background: #1A1D27;
        }}

        .compare-table tbody tr:nth-child(even) {{
            background: #1F2333;
        }}

        .compare-table tbody tr:hover {{
            background: rgba(99, 102, 241, 0.14);
        }}

        @keyframes predictPulse {{
            0%, 100% {{ box-shadow: 0 0 0 1px rgba(99,102,241,0.45), 0 0 16px rgba(99,102,241,0.2); }}
            50% {{ box-shadow: 0 0 0 1px rgba(99,102,241,0.6), 0 0 28px rgba(99,102,241,0.45); }}
        }}

        @keyframes pulseDot {{
            0% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }}
            70% {{ box-shadow: 0 0 0 8px rgba(34,197,94,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }}
        }}

        @keyframes shimmer {{
            from {{ background-position: 0% 50%; }}
            to {{ background-position: 200% 50%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state(features: list[str]) -> None:
    """Initialize Streamlit session state variables used across the app."""
    # Initialize prediction output state once so widget changes do not clear results.
    if "predicted_price" not in st.session_state:
        st.session_state.predicted_price = None

    # Initialize explanation text state once for persistence.
    if "explanation" not in st.session_state:
        st.session_state.explanation = ""

    # Initialize SHAP value state once for persistence.
    if "shap_values" not in st.session_state:
        st.session_state.shap_values = None

    # Initialize last-used input feature dictionary for replayable context.
    if "last_input" not in st.session_state:
        st.session_state.last_input = {feature: 0.0 for feature in features}

    # Store confidence interval outputs for the latest prediction.
    if "prediction_low" not in st.session_state:
        st.session_state.prediction_low = None

    if "prediction_high" not in st.session_state:
        st.session_state.prediction_high = None

    if "prediction_std" not in st.session_state:
        st.session_state.prediction_std = None

    # Track active preset chip.
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = ""

    # Track brief prediction animation/feedback state.
    if "last_prediction_ts" not in st.session_state:
        st.session_state.last_prediction_ts = None

    # Keep auto-predict enabled after the first prediction.
    if "auto_predict_enabled" not in st.session_state:
        st.session_state.auto_predict_enabled = True


def sanitize_explanation_for_display(text: str, features_dict: dict, predicted_price: float) -> str:
    """Convert meta/instructional AI output into a short user-facing sentence."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lowered = cleaned.lower()
    meta_starts = (
        "i need to explain",
        "let's analyze",
        "analysis:",
        "instructions:",
        "here is my reasoning",
    )
    if lowered.startswith(meta_starts):
        area = features_dict.get("GrLivArea", "the home's size")
        beds = features_dict.get("BedroomAbvGr", "the bedroom count")
        baths = features_dict.get("FullBath", "the bathroom count")
        year = features_dict.get("YearBuilt", "the build year")
        garage = features_dict.get("GarageArea", "the garage size")
        return (
            f"This home is estimated at ${predicted_price:,.0f} mainly due to its {area} sq ft size, "
            f"{beds} bedrooms, {baths} baths, {year} build year, and {garage} sq ft garage."
        )

    first_line = cleaned.splitlines()[0].strip()
    return first_line


def build_input_dataframe(input_values: dict, feature_order: list[str]) -> pd.DataFrame:
    """Build a one-row DataFrame in model feature order from input values."""
    # Create one-row DataFrame using the trained feature order for prediction safety.
    return pd.DataFrame([[input_values[feature] for feature in feature_order]], columns=feature_order)


def predict_price(model, input_df: pd.DataFrame) -> float:
    """Predict price from a one-row input DataFrame and return a float."""
    # Call the trained regressor and convert the scalar prediction to float.
    prediction = model.predict(input_df)[0]
    return float(prediction)


def predict_price_with_confidence(model, input_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Predict price and return an approximate confidence interval from tree spread."""
    prediction = predict_price(model, input_df)

    if hasattr(model, "estimators_") and model.estimators_:
        tree_predictions = np.array([float(tree.predict(input_df)[0]) for tree in model.estimators_], dtype=float)
        low = float(np.percentile(tree_predictions, 10))
        high = float(np.percentile(tree_predictions, 90))
        std = float(np.std(tree_predictions))
        return prediction, low, high, std

    fallback_margin = max(abs(prediction) * 0.1, 10_000.0)
    return prediction, prediction - fallback_margin, prediction + fallback_margin, fallback_margin / 2


def run_prediction_pipeline(input_values: dict, feature_order: list[str], model) -> None:
    """Run prediction + SHAP + explanation and persist in session state."""
    input_df = build_input_dataframe(input_values, feature_order)
    predicted_price, pred_low, pred_high, pred_std = predict_price_with_confidence(model, input_df)
    st.session_state.predicted_price = predicted_price
    st.session_state.prediction_low = pred_low
    st.session_state.prediction_high = pred_high
    st.session_state.prediction_std = pred_std
    st.session_state.last_input = input_values.copy()
    st.session_state.last_prediction_ts = time.time()

    try:
        st.session_state.shap_values = get_shap_values(input_df)
    except Exception:
        st.session_state.shap_values = None

    raw_explanation = explain_prediction(input_values, predicted_price)
    st.session_state.explanation = sanitize_explanation_for_display(
        raw_explanation,
        input_values,
        predicted_price,
    )


def get_feature_bounds(dataset: pd.DataFrame, feature_order: list[str]) -> dict:
    """Return data-driven min/max/default bounds for each feature slider."""
    bounds = {}
    for feature in feature_order:
        if feature not in dataset.columns:
            raise ValueError(f"Required feature column is missing from dataset: {feature}")
        series = dataset[feature].dropna()
        min_value = int(series.min()) if not series.empty else 0
        max_value = int(series.max()) if not series.empty else 100
        default_value = int(series.median()) if not series.empty else 0
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        if default_value < min_value or default_value > max_value:
            default_value = int(np.clip(default_value, min_value, max_value))
        bounds[feature] = (min_value, max_value, default_value)
    return bounds


def apply_slider_preset(preset_name: str, feature_bounds: dict) -> None:
    """Apply a quick preset to slider session-state values."""
    updated_values = {}
    for feature, (_, _, default_value) in feature_bounds.items():
        slider_key = f"slider_{feature}"
        updated_values[feature] = int(st.session_state.get(slider_key, default_value))

    presets = {
        "Starter Home": {
            "GrLivArea": 900,
            "BedroomAbvGr": 2,
            "FullBath": 1,
            "YearBuilt": 1955,
            "GarageArea": 200,
        },
        "Family Home": {
            "GrLivArea": 1464,
            "BedroomAbvGr": 3,
            "FullBath": 2,
            "YearBuilt": 1973,
            "GarageArea": 480,
        },
        "Luxury Home": {
            "GrLivArea": 3200,
            "BedroomAbvGr": 5,
            "FullBath": 4,
            "YearBuilt": 2005,
            "GarageArea": 900,
        },
    }

    if preset_name in presets:
        updated_values.update(presets[preset_name])

    for feature, (min_value, max_value, _) in feature_bounds.items():
        slider_key = f"slider_{feature}"
        clipped = int(np.clip(updated_values.get(feature, min_value), min_value, max_value))
        st.session_state[slider_key] = clipped


def find_comparable_homes(input_values: dict, source_df: pd.DataFrame, feature_order: list[str], top_k: int = 5) -> pd.DataFrame:
    """Find similar homes in the dataset using normalized feature distance."""
    if top_k <= 0:
        return pd.DataFrame()
    if "SalePrice" not in source_df.columns:
        return pd.DataFrame()
    missing_columns = [feature for feature in feature_order if feature not in source_df.columns]
    if missing_columns:
        return pd.DataFrame()

    comparable_df = source_df[feature_order + ["SalePrice"]].dropna().copy()
    if comparable_df.empty:
        return comparable_df

    x = comparable_df[feature_order].astype(float)
    means = x.mean()
    stds = x.std(ddof=0).replace(0, 1)

    target = pd.Series({feature: float(input_values.get(feature, means[feature])) for feature in feature_order})
    normalized_x = (x - means) / stds
    normalized_target = (target - means) / stds
    comparable_df["SimilarityScore"] = np.sqrt(((normalized_x - normalized_target) ** 2).sum(axis=1))

    nearest = comparable_df.nsmallest(top_k, "SimilarityScore").copy()
    nearest["SalePrice"] = nearest["SalePrice"].astype(float)
    return nearest.reset_index(drop=True)


def _format_signed_currency(value: float) -> str:
    """Format a signed value as currency with explicit sign."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.0f}"


def get_feature_price_slopes(source_df: pd.DataFrame, feature_order: list[str]) -> dict[str, float]:
    """Estimate per-feature dollar slope for lightweight slider delta hints."""
    if "SalePrice" not in source_df.columns:
        return {feature: 0.0 for feature in feature_order}

    training_df = source_df[feature_order + ["SalePrice"]].dropna().copy()
    if training_df.empty:
        return {feature: 0.0 for feature in feature_order}

    sale = training_df["SalePrice"].astype(float)
    slopes: dict[str, float] = {}
    for feature in feature_order:
        x = training_df[feature].astype(float)
        var = float(x.var())
        if var <= 1e-9:
            slopes[feature] = 0.0
            continue
        slopes[feature] = float(np.cov(x, sale, ddof=0)[0, 1] / var)
    return slopes


def build_feature_impact_chart(shap_values, feature_names) -> tuple[go.Figure, pd.DataFrame]:
    """Build minimal horizontal feature impact chart sorted by absolute impact."""
    values = np.array(shap_values, dtype=float).reshape(-1)
    if len(values) != len(feature_names):
        raise ValueError("SHAP values and feature names must have matching lengths.")
    impacts = pd.DataFrame({"Feature": feature_names, "Impact": values})
    impacts["AbsImpact"] = impacts["Impact"].abs()
    impacts = impacts.sort_values("AbsImpact", ascending=False).drop(columns=["AbsImpact"])
    colors = ["#1D9E75" if value >= 0 else "#E24B4A" for value in impacts["Impact"]]
    dollar_labels = [_format_signed_currency(value) for value in impacts["Impact"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=impacts["Impact"],
                y=impacts["Feature"],
                orientation="h",
                marker_color=colors,
                text=dollar_labels,
                textposition="outside",
                hoverlabel=dict(bgcolor="#4F46E5", font_color="#FFFFFF"),
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1A1D27",
        font_color="#8B92A5",
        xaxis=dict(gridcolor="#2A2D3E", zerolinecolor="#2A2D3E"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig, impacts


def estimate_quick_delta(current_input: dict, previous_input: dict, source_df: pd.DataFrame, feature_order: list[str]) -> float:
    """Estimate rough delta before re-predicting using per-feature linear slopes."""
    if "SalePrice" not in source_df.columns:
        return 0.0

    training_df = source_df[feature_order + ["SalePrice"]].dropna().copy()
    if training_df.empty:
        return 0.0

    delta = 0.0
    sale = training_df["SalePrice"].astype(float)
    for feature in feature_order:
        x = training_df[feature].astype(float)
        var = float(x.var())
        if var <= 1e-9:
            continue
        slope = float(np.cov(x, sale, ddof=0)[0, 1] / var)
        delta += slope * (float(current_input[feature]) - float(previous_input.get(feature, current_input[feature])))
    return float(delta)


@st.cache_resource
def train_comparison_models(source_df: pd.DataFrame, feature_order: tuple[str, ...]) -> dict[str, object]:
    """Train comparison models once for real-time side-by-side prediction checks."""
    required_columns = list(feature_order) + ["SalePrice"]
    training_df = source_df[required_columns].dropna().copy()
    x_train = training_df[list(feature_order)].astype(float)
    y_train = training_df["SalePrice"].astype(float)

    comparison_models: dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBRegressor": XGBRegressor(n_estimators=200, random_state=42, eval_metric="rmse", verbosity=0),
    }
    for candidate_model in comparison_models.values():
        candidate_model.fit(x_train, y_train)
    return comparison_models


def get_model_display_name(model_key: str) -> str:
    """Map artifact model keys to concise display labels."""
    mapping = {
        "LinearRegression": "Linear Regression",
        "RandomForestRegressor": "Random Forest",
        "XGBRegressor": "XGBoost",
    }
    return mapping.get(model_key, model_key)


def build_confidence_breakdown_chart(predicted: float, lower: float, upper: float, std_dev: float) -> go.Figure:
    """Build a compact horizontal bar summary for confidence components."""
    labels = ["Lower Bound", "Predicted Price", "Upper Bound", "Std Deviation"]
    values = [float(lower), float(predicted), float(upper), float(std_dev)]
    colors = ["#4B5563", "#7F77DD", "#4B5563", "#1D9E75"]

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=colors,
                text=[f"${value:,.0f}" for value in values],
                textposition="outside",
                hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1A1D27",
        font_color="#8B92A5",
        xaxis=dict(gridcolor="#2A2D3E", zerolinecolor="#2A2D3E", title="USD"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
    )
    return fig


def build_stepper_markup(has_prediction: bool) -> str:
    """Return dynamic stepper markup with completed and active states."""
    if has_prediction:
        return (
            '<div class="stepper">'
            '<span class="step completed"><span class="icon">&#10003;</span>1 Input</span>'
            '<span class="step-sep">&bull;</span>'
            '<span class="step completed"><span class="icon">&#10003;</span>2 Predict</span>'
            '<span class="step-sep">&bull;</span>'
            '<span class="step active"><span class="icon">&#9673;</span>3 Analyze</span>'
            '</div>'
        )
    return (
        '<div class="stepper">'
        '<span class="step active"><span class="icon">&#9673;</span>1 Input</span>'
        '<span class="step-sep">&bull;</span>'
        '<span class="step"><span class="icon">&#9678;</span>2 Predict</span>'
        '<span class="step-sep">&bull;</span>'
        '<span class="step"><span class="icon">&#9678;</span>3 Analyze</span>'
        '</div>'
    )


# Load core artifacts safely before rendering UI sections.
model = safe_load_joblib(MODEL_PATH, "trained model")
features = safe_load_joblib(FEATURES_PATH, "feature list")
metrics_artifact = safe_load_joblib(METRICS_PATH, "metrics")

if model is None:
    st.error("Trained model could not be loaded.")
    st.stop()

# Load dataset for slider ranges and robust defaults.
dataset = safe_load_data(DATA_PATH)
try:
    feature_bounds = get_feature_bounds(dataset, features)
except Exception as exc:
    st.error(f"Failed to compute feature bounds from dataset: {exc}")
    st.stop()

# Initialize all persistent app state values.
initialize_state(features)
apply_monochrome_theme(has_prediction=st.session_state.predicted_price is not None)

with st.sidebar:
    stepper_markup = build_stepper_markup(st.session_state.predicted_price is not None)
    st.markdown(
        f"""
        <p class="sidebar-top-title">Valora AI</p>
        <p class="brand-tag">Model-driven real estate valuation</p>
        {stepper_markup}
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="muted-label">Preset Scenarios</p>', unsafe_allow_html=True)
    with st.container(key="preset_chip_group"):
        if st.button(
            "Starter Home  Compact older build",
            key="preset_chip_starter",
            type="primary" if st.session_state.active_preset == "Starter Home" else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_preset = "Starter Home"
            apply_slider_preset("Starter Home", feature_bounds)
        if st.button(
            "Family Home  Balanced everyday layout",
            key="preset_chip_family",
            type="primary" if st.session_state.active_preset == "Family Home" else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_preset = "Family Home"
            apply_slider_preset("Family Home", feature_bounds)
        if st.button(
            "Luxury Home  Large modern profile",
            key="preset_chip_luxury",
            type="primary" if st.session_state.active_preset == "Luxury Home" else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_preset = "Luxury Home"
            apply_slider_preset("Luxury Home", feature_bounds)

    st.divider()
    st.markdown('<p class="muted-label">Property Features</p>', unsafe_allow_html=True)

    current_input = {}
    feature_slopes = get_feature_price_slopes(dataset, features)
    feature_display_names = {
        "GrLivArea": "Living area",
        "BedroomAbvGr": "Bedrooms",
        "FullBath": "Bathrooms",
        "YearBuilt": "Year built",
        "GarageArea": "Garage area",
    }
    for feature in features:
        min_value, max_value, default_value = feature_bounds[feature]
        slider_value_slot = st.empty()
        current_input[feature] = st.slider(
            label=feature,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            key=f"slider_{feature}",
            label_visibility="collapsed",
        )
        slider_value_slot.markdown(
            (
                "<div class='slider-value-row'>"
                f"<span class='name'>{html.escape(feature_display_names.get(feature) or str(feature))}</span>"
                f"<span class='value'>{int(current_input[feature]):,}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        baseline_value = float(st.session_state.last_input.get(feature, current_input[feature]))
        delta_units = float(current_input[feature]) - baseline_value
        est_delta = float(feature_slopes.get(feature, 0.0) * delta_units)
        if st.session_state.predicted_price is not None and abs(est_delta) >= 1.0:
            delta_class = "pos" if est_delta >= 0 else "neg"
            st.markdown(
                f"<span class='slider-delta-chip {delta_class}'>{_format_signed_currency(est_delta)} est.</span>",
                unsafe_allow_html=True,
            )

    if st.session_state.predicted_price is not None:
        estimated_delta = estimate_quick_delta(current_input, st.session_state.last_input, dataset, features)
        if abs(estimated_delta) >= 1.0:
            delta_arrow = "+" if estimated_delta >= 0 else "-"
            delta_color = "#059669" if estimated_delta >= 0 else "#DC2626"
            st.markdown(
                f"<p style='margin:0.25rem 0 0.5rem;font-size:12px;color:{delta_color};font-weight:600;'>"
                f"{delta_arrow} ${abs(estimated_delta):,.0f} estimated total change</p>",
                unsafe_allow_html=True,
            )

    st.checkbox("Auto-predict after slider changes", key="auto_predict_enabled")

    predict_feedback = st.empty()
    with st.container(key="predict_btn_wrap"):
        predict_clicked = st.button("Predict Price", key="predict_price_button", use_container_width=True)

inputs_changed_since_last_prediction = any(
    float(current_input[feature]) != float(st.session_state.last_input.get(feature, current_input[feature]))
    for feature in features
)

auto_predict_trigger = bool(
    st.session_state.auto_predict_enabled
    and st.session_state.predicted_price is not None
    and inputs_changed_since_last_prediction
)

if predict_clicked or auto_predict_trigger:
    spinner_text = "Computing valuation..." if predict_clicked else "Refreshing estimate..."
    with st.spinner(spinner_text):
        run_prediction_pipeline(current_input, features, model)

    if predict_clicked:
        predict_feedback.markdown(
            "<div class='prediction-ready-badge'><span class='dot'></span><span>Prediction Ready</span></div>",
            unsafe_allow_html=True,
        )
        time.sleep(1)

active_input = st.session_state.last_input if st.session_state.predicted_price is not None else current_input

left_col, right_col = st.columns([1.2, 1])

with left_col:
    with st.container(key="card_property_summary"):
        st.markdown('<p class="muted-label">Property Summary</p>', unsafe_allow_html=True)
        summary_col1, summary_col2 = st.columns(2)
        summary_col1.metric("Living Area", f"{int(active_input.get('GrLivArea', 0)):,} sq ft")
        summary_col2.metric("Garage", f"{int(active_input.get('GarageArea', 0)):,} sq ft")
        summary_col3, summary_col4 = st.columns(2)
        summary_col3.metric("Bedrooms", f"{int(active_input.get('BedroomAbvGr', 0))}")
        summary_col4.metric("Bathrooms", f"{int(active_input.get('FullBath', 0))}")
        st.metric("Year Built", f"{int(active_input.get('YearBuilt', 0))}")

    with st.container(key="card_feature_impact"):
        st.markdown('<p class="muted-label">Feature Impact</p>', unsafe_allow_html=True)
        if st.session_state.shap_values is not None:
            impact_fig, _ = build_feature_impact_chart(st.session_state.shap_values, features)
            st.plotly_chart(impact_fig, use_container_width=True)
        else:
            st.info("Run a prediction to view feature impact.")

    with st.container(key="card_realtime_model_comparison"):
        st.markdown('<p class="muted-label">Real-Time Model Comparison</p>', unsafe_allow_html=True)
        comparison_models = train_comparison_models(dataset, tuple(features))
        comparison_input_df = build_input_dataframe(current_input, features)
        comparison_predictions = {
            "LinearRegression": float(comparison_models["LinearRegression"].predict(comparison_input_df)[0]),
            "RandomForestRegressor": float(model.predict(comparison_input_df)[0]),
            "XGBRegressor": float(comparison_models["XGBRegressor"].predict(comparison_input_df)[0]),
        }

        model_color_map = {
            "LinearRegression": "#4B5563",
            "RandomForestRegressor": "#7F77DD",
            "XGBRegressor": "#1D9E75",
        }
        comparison_chart = go.Figure()
        for model_key in ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]:
            comparison_chart.add_trace(
                go.Bar(
                    name=get_model_display_name(model_key),
                    x=["Current Input"],
                    y=[comparison_predictions[model_key]],
                    marker_color=model_color_map[model_key],
                    text=[f"${comparison_predictions[model_key]:,.0f}"],
                    textposition="outside",
                    hovertemplate=f"{get_model_display_name(model_key)}: $%{{y:,.0f}}<extra></extra>",
                )
            )
        comparison_chart.update_layout(
            barmode="group",
            height=330,
            margin=dict(l=10, r=10, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1A1D27",
            font_color="#8B92A5",
            yaxis=dict(title="Predicted Price (USD)", gridcolor="#2A2D3E", zerolinecolor="#2A2D3E"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            legend_title_text="Model",
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
        st.caption("Production model: Random Forest")

        if isinstance(metrics_artifact, dict):
            metrics_rows = []
            for model_key in ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]:
                model_metrics = metrics_artifact.get(model_key, {})
                metrics_rows.append(
                    {
                        "Model": get_model_display_name(model_key),
                        "Role": "Production" if model_key == "RandomForestRegressor" else "Candidate",
                        "R2": float(model_metrics.get("R2", np.nan)),
                        "RMSE": float(model_metrics.get("RMSE", np.nan)),
                        "MAE": float(model_metrics.get("MAE", np.nan)),
                    }
                )
            metrics_df = pd.DataFrame(metrics_rows)
            styled_metrics_df = metrics_df.style.format(
                {"R2": "{:.4f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}"}
            ).apply(
                lambda row: [
                    "background-color: rgba(127,119,221,0.16); font-weight: 600;" if row["Role"] == "Production" else ""
                    for _ in row
                ],
                axis=1,
            )
            st.dataframe(styled_metrics_df, use_container_width=True, hide_index=True)

with right_col:
    with st.container(key="card_prediction_output"):
        st.markdown('<p class="muted-label">Prediction Output</p>', unsafe_allow_html=True)
        if st.session_state.predicted_price is None:
            st.info("Set feature values and click Predict Price to generate an estimate.")
        else:
            pred = float(st.session_state.predicted_price)
            low = float(st.session_state.prediction_low) if st.session_state.prediction_low is not None else pred * 0.9
            high = float(st.session_state.prediction_high) if st.session_state.prediction_high is not None else pred * 1.1
            std = float(st.session_state.prediction_std) if st.session_state.prediction_std is not None else max(abs(pred) * 0.05, 1.0)
            confidence_score = float(np.clip(1.0 - (std / max(abs(pred), 1.0)), 0.05, 0.98))

            price_slot = st.empty()
            range_text = f"Confidence range: ${low/1000:,.0f}K - ${high/1000:,.0f}K"
            animate_now = st.session_state.last_prediction_ts is not None
            if animate_now:
                for step in range(1, 21):
                    animated_price = pred * (step / 20)
                    price_slot.markdown(
                        f"""
                        <div class="prediction-value-card">
                          <p class="k">Estimated Value</p>
                          <p class="v">${animated_price:,.0f}</p>
                          <p class="r">{range_text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.025)
                st.session_state.last_prediction_ts = None
            else:
                price_slot.markdown(
                    f"""
                    <div class="prediction-value-card">
                      <p class="k">Estimated Value</p>
                      <p class="v">${pred:,.0f}</p>
                      <p class="r">{range_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            confidence_pct = int(confidence_score * 100)
            st.markdown(
                f"""
                <div class="confidence-wrap">
                  <div class="confidence-meta">
                    <span>Confidence Score</span>
                    <span>{confidence_pct}%</span>
                  </div>
                  <div class="confidence-track">
                    <div class="confidence-fill" style="width:{confidence_pct}%;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Sq Ft", f"{int(st.session_state.last_input.get('GrLivArea', 0)):,}")
            m2.metric("Year Built", f"{int(st.session_state.last_input.get('YearBuilt', 0))}")
            m3.metric(
                "Beds/Baths",
                f"{int(st.session_state.last_input.get('BedroomAbvGr', 0))}/{int(st.session_state.last_input.get('FullBath', 0))}",
            )

            if st.session_state.explanation:
                st.markdown(f'<div class="chat-bubble">{html.escape(st.session_state.explanation)}</div>', unsafe_allow_html=True)

    with st.container(key="card_confidence_breakdown"):
        st.markdown('<p class="muted-label">Confidence Breakdown</p>', unsafe_allow_html=True)
        if st.session_state.predicted_price is None:
            st.info("Generate a prediction to see confidence breakdown details.")
        else:
            conf_pred = float(st.session_state.predicted_price)
            conf_low = (
                float(st.session_state.prediction_low)
                if st.session_state.prediction_low is not None
                else conf_pred * 0.9
            )
            conf_high = (
                float(st.session_state.prediction_high)
                if st.session_state.prediction_high is not None
                else conf_pred * 1.1
            )
            conf_std = (
                float(st.session_state.prediction_std)
                if st.session_state.prediction_std is not None
                else max(abs(conf_pred) * 0.05, 1.0)
            )
            confidence_breakdown_fig = build_confidence_breakdown_chart(
                predicted=conf_pred,
                lower=conf_low,
                upper=conf_high,
                std_dev=conf_std,
            )
            st.plotly_chart(confidence_breakdown_fig, use_container_width=True)

    with st.container(key="card_comparable_homes"):
        st.markdown('<p class="muted-label">Comparable Homes</p>', unsafe_allow_html=True)
        if st.session_state.predicted_price is None:
            st.info("Generate a prediction to see similar homes.")
        else:
            comparables = find_comparable_homes(st.session_state.last_input, dataset, features, top_k=3)
            if comparables.empty:
                st.info("Comparable homes are unavailable for the current data.")
            else:
                mini = pd.DataFrame(
                    {
                        "Sq Ft": comparables["GrLivArea"].astype(int),
                        "Beds": comparables["BedroomAbvGr"].astype(int),
                        "Year": comparables["YearBuilt"].astype(int),
                        "Est. Price": comparables["SalePrice"].astype(float).map(lambda x: f"${x:,.0f}"),
                    }
                ).reset_index(drop=True)
                mini_html = mini.to_html(index=False, classes="compare-table", border=0)
                st.markdown(
                    f"<div class='compare-table-wrap'>{mini_html}</div>",
                    unsafe_allow_html=True,
                )




