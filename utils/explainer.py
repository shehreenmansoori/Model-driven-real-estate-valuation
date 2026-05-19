"""Utilities for SHAP explanations and Plotly visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "saved_model.pkl"


_model: RandomForestRegressor | None = None
_explainer: Any | None = None


def _load() -> Any:
    """Load and cache model + explainer only once."""
    global _model
    global _explainer

    if _explainer is None:
        _model = joblib.load(MODEL_PATH)
        if not isinstance(_model, RandomForestRegressor):
            raise TypeError("Saved model must be a RandomForestRegressor for SHAP explanations.")
        _explainer = shap.TreeExplainer(_model)

    if _explainer is None:
        raise RuntimeError("Failed to initialize SHAP explainer.")
    return _explainer


def get_shap_values(input_df: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for the given input DataFrame using TreeExplainer."""
    # Build or reuse a cached tree explainer for the RandomForest model.
    explainer = _load()

    # Compute SHAP values for the provided input row(s).
    shap_vals = explainer.shap_values(input_df)
    # Handle both ndarray output and legacy list output shapes.
    if isinstance(shap_vals, list):
        if not shap_vals:
            raise ValueError("SHAP returned an empty values list.")
        first = np.array(shap_vals[0], dtype=float)
    else:
        first = np.array(shap_vals, dtype=float)

    if first.ndim == 1:
        return first
    return first[0]


def get_shap_plotly_chart(shap_values, feature_names):
    """Create a horizontal Plotly bar chart for SHAP feature contributions."""
    # Convert SHAP output to a 1D feature-level vector for plotting.
    values_array = np.array(shap_values)
    if values_array.ndim == 1:
        feature_impacts = values_array
    else:
        feature_impacts = values_array.reshape(-1)

    # Ensure SHAP vector and feature names align before plotting.
    if len(feature_impacts) != len(feature_names):
        raise ValueError("SHAP values and feature names must have matching lengths.")

    # Choose green for positive and red for negative feature impacts.
    colors = ["#16a34a" if value >= 0 else "#dc2626" for value in feature_impacts]

    # Build a horizontal bar chart to show contribution directions clearly.
    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_impacts,
                y=feature_names,
                orientation="h",
                marker_color=colors,
                text=[f"{value:,.2f}" for value in feature_impacts],
                textposition="auto",
            )
        ]
    )

    # Final chart layout with the exact title requested.
    fig.update_layout(
        title="Why did the model predict this price?",
        xaxis_title="SHAP Value (Contribution to Price)",
        yaxis_title="Feature",
        template="plotly_white",
        height=360,
    )
    return fig
