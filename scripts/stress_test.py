"""Project stress checks for model, explainability, and API parsing helpers."""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_project_guide_pdf import wrap_markdown_lines
from model.train_model import FEATURES, TARGET, load_training_data
from utils import explainer
from utils import openrouter_api as api

DATA_PATH = ROOT / "data" / "house_prices.csv"
MODEL_PATH = ROOT / "model" / "saved_model.pkl"
FEATURES_PATH = ROOT / "model" / "features.pkl"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_prediction_stress(iterations: int = 5000) -> None:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    dataset = pd.read_csv(DATA_PATH)

    _assert(features == FEATURES, "Saved features artifact does not match training FEATURES constant.")
    _assert(all(feature in dataset.columns for feature in features), "Dataset is missing one or more required features.")

    bounds: dict[str, tuple[float, float]] = {}
    for feature in features:
        series = dataset[feature].dropna().astype(float)
        bounds[feature] = (float(series.min()), float(series.max()))

    for i in range(iterations):
        row = {feature: random.uniform(*bounds[feature]) for feature in features}
        input_df = pd.DataFrame([[row[feature] for feature in features]], columns=features)
        prediction = float(model.predict(input_df)[0])
        _assert(np.isfinite(prediction), f"Prediction was not finite at iteration {i}: {prediction}")

    print(f"[PASS] prediction stress: {iterations} randomized inputs")


def run_shap_stress(sample_size: int = 25) -> None:
    dataset = pd.read_csv(DATA_PATH)
    sample = dataset[FEATURES].dropna().sample(n=sample_size, random_state=42)

    for i, (_, row) in enumerate(sample.iterrows()):
        input_df = pd.DataFrame([row.values], columns=FEATURES)
        values = np.array(explainer.get_shap_values(input_df), dtype=float).reshape(-1)
        _assert(
            len(values) == len(FEATURES),
            f"SHAP vector length mismatch at sample {i}: expected {len(FEATURES)}, got {len(values)}",
        )
        _assert(np.isfinite(values).all(), f"Non-finite SHAP values at sample {i}")

    print(f"[PASS] SHAP stress: {sample_size} sampled rows")


def run_openrouter_parser_stress() -> None:
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.text = str(payload)

        def json(self):
            return self._payload

    payloads = [
        {"choices": [{"message": {"content": "plain text"}}]},
        {"choices": [{"message": {"content": [{"text": "chunk one"}, {"text": "chunk two"}]}}]},
        {"choices": [{"text": "fallback text field"}]},
    ]

    for payload in payloads:
        extracted = api._extract_response_content(FakeResponse(payload))
        _assert(bool(extracted.strip()), "Parser returned empty text")

    print(f"[PASS] OpenRouter parser stress: {len(payloads)} response shapes")


def run_data_loader_edge_checks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "bad.csv"
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(temp_path, index=False)
        try:
            load_training_data(temp_path)
        except ValueError:
            pass
        else:
            raise AssertionError("load_training_data accepted invalid schema unexpectedly.")

    try:
        load_training_data(ROOT / "data" / "__does_not_exist__.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("load_training_data did not fail for missing file.")

    print("[PASS] training data loader edge checks")


def run_markdown_wrap_stress() -> None:
    text = (
        "# Title\n"
        "1. This is a very long numbered item that should wrap correctly and preserve indentation "
        "for continuation lines without losing readability.\n"
        "- Another very long bullet that should still keep bullet formatting even when wrapping "
        "beyond the configured line width.\n"
        "```\nprint('code block should not wrap')\n```\n"
    )
    wrapped = wrap_markdown_lines(text, width=50)
    _assert(any(line.startswith("1. ") for line in wrapped), "Numbered bullet prefix was not preserved.")
    _assert(any(line.startswith("- ") for line in wrapped), "Dash bullet prefix was not preserved.")
    _assert(any(line.startswith("```") for line in wrapped), "Code fence marker was not preserved.")

    print("[PASS] markdown wrapping stress")


def main() -> None:
    print("Running project stress suite...")
    run_prediction_stress()
    run_shap_stress()
    run_openrouter_parser_stress()
    run_data_loader_edge_checks()
    run_markdown_wrap_stress()
    print("All stress checks passed.")


if __name__ == "__main__":
    main()
