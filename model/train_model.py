"""Train and evaluate house price prediction models."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _assert_supported_python() -> None:
    """Fail fast with a clear message on unsupported Python versions."""
    version = sys.version_info[:2]
    if version < (3, 11) or version >= (3, 13):
        raise RuntimeError(
            f"Unsupported Python {sys.version.split()[0]}. "
            "Use Python 3.11 or 3.12 for this project."
        )


_assert_supported_python()

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt", "GarageArea"]
TARGET = "SalePrice"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_training_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean only the columns needed for model training."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at: {csv_path}")

    # Read the source CSV and keep only required columns.
    required_columns = FEATURES + [TARGET]
    try:
        df = pd.read_csv(csv_path, usecols=required_columns)
    except ValueError as exc:
        raise ValueError(
            "Dataset schema mismatch. Ensure house_prices.csv contains required columns: "
            + ", ".join(required_columns)
        ) from exc

    # Drop rows with missing values in just the required six columns.
    cleaned_df = df.dropna(subset=required_columns).copy()
    if cleaned_df.empty:
        raise ValueError("Training data is empty after dropping rows with missing required values.")
    return cleaned_df


def evaluate_model(name: str, model, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    """Train a model and return evaluation metrics for that model."""
    # Fit the model on the training split.
    model.fit(x_train, y_train)

    # Create predictions on the test split for evaluation.
    predictions = model.predict(x_test)

    # Compute requested regression metrics.
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "Model": name,
        "R2": float(r2_score(y_test, predictions)),
        "RMSE": float(mse**0.5),
        "MAE": float(mean_absolute_error(y_test, predictions)),
    }
    return metrics


def print_summary_table(metrics_dict: dict) -> None:
    """Print a clean summary table for all model scores."""
    # Convert metrics dictionary to a tabular DataFrame for pretty printing.
    rows = []
    for model_name, scores in metrics_dict.items():
        rows.append(
            {
                "Model": model_name,
                "R2": scores["R2"],
                "RMSE": scores["RMSE"],
                "MAE": scores["MAE"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

    # Print the final formatted table to terminal.
    print("\nModel Performance Summary")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))


def main() -> None:
    """Run training, evaluation, and artifact export for the project."""
    # Resolve important project paths.
    data_path = Path(BASE_DIR) / "data" / "house_prices.csv"
    model_dir = Path(BASE_DIR) / "model"

    # Ensure the output directory exists before writing artifacts.
    os.makedirs(model_dir, exist_ok=True)

    # Load and clean the training dataset.
    dataset = load_training_data(data_path)
    x = dataset[FEATURES]
    y = dataset[TARGET]
    if len(dataset) < 10:
        raise ValueError(
            f"Training dataset is too small ({len(dataset)} rows). "
            "Provide at least 10 clean rows for stable training."
        )

    # Split data into training and test partitions.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Define all models requested for comparison.
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBRegressor": XGBRegressor(n_estimators=200, random_state=42, eval_metric="rmse", verbosity=0),
    }

    # Train and evaluate each model, storing scores in a metrics dictionary.
    metrics = {}
    for model_name, model in models.items():
        metrics[model_name] = evaluate_model(model_name, model, x_train, x_test, y_train, y_test)

    # Train the final production model (RandomForestRegressor) on all cleaned data.
    final_model = RandomForestRegressor(n_estimators=200, random_state=42)
    final_model.fit(x, y)

    # Save final model and metadata artifacts to disk.
    joblib.dump(final_model, model_dir / "saved_model.pkl")
    joblib.dump(metrics, model_dir / "metrics.pkl")
    joblib.dump(FEATURES, model_dir / "features.pkl")

    # Print a clean table so users can review model performance quickly.
    print_summary_table(metrics)


def train() -> None:
    """Run the project training pipeline."""
    main()


if __name__ == "__main__":
    # Execute the full training workflow when this script is run directly.
    train()
