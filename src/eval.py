import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Paths
PREPROCESSED_DIR = "data/preprocessed"
TEST_FILE = os.path.join(PREPROCESSED_DIR, "test.csv")
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SLIDING_WINDOW = 30
NUM_CURRENCIES = 8
FORECAST_HORIZON = 7
CURRENCIES = ["AUD", "GBP", "CAD", "CHF", "CNY", "JPY", "NZD", "SGD"]

class TransformerForecastingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, forecast_horizon, num_currencies):
        super().__init__()

        # Currency embedding
        self.currency_embedding = torch.nn.Linear(input_dim, hidden_dim)

        # Transformer
        self.transformer = torch.nn.Transformer(
            d_model=hidden_dim, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True
        )

        # Decoder
        self.decoder = torch.nn.Linear(hidden_dim, forecast_horizon * num_currencies)

    def forward(self, x):
        # Apply currency embedding
        x = self.currency_embedding(x)

        # Transformer
        x = self.transformer(x, x)

        # Decode to output shape: [batch_size, forecast_horizon, num_currencies]
        x = self.decoder(x[:, -1, :])
        return x.view(x.size(0), FORECAST_HORIZON, NUM_CURRENCIES)

def load_model(model_path, config_path):
    """Load a trained model and its configuration."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model = TransformerForecastingModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        forecast_horizon=config["forecast_horizon"],
        num_currencies=config["num_currencies"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_test_data(test_df, sliding_window):
    """Prepare test data for prediction."""
    x, y = [], []
    for i in range(len(test_df) - sliding_window - FORECAST_HORIZON + 1):
        x.append(test_df.iloc[i:i + sliding_window].values)
        y.append(test_df.iloc[i + sliding_window:i + sliding_window + FORECAST_HORIZON].values)
    return torch.tensor(np.array(x), dtype=torch.float32), np.array(y)

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    metrics = {}
    metrics["MSE"] = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    metrics["MAE"] = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)
    metrics["R^2"] = r2_score(y_true, y_pred, multioutput="raw_values")
    return metrics

def evaluate_model():
    """Evaluate the model on the test set and save results."""
    print("Loading test data...")
    test_df = pd.read_csv(TEST_FILE)
    test_data, test_targets = prepare_test_data(test_df, sliding_window=SLIDING_WINDOW)

    print("Loading model...")
    model_path = os.path.join(CONFIGS_DIR, "transformer_model_7_day.pth")
    config_path = model_path.replace(".pth", "_config.json")
    model = load_model(model_path, config_path)

    print("Generating predictions...")
    predictions = model(test_data).detach().numpy()

    for day in range(1, FORECAST_HORIZON + 1):
        print(f"Evaluating metrics for {day}-day forecast...")
        observed = test_targets[:, day - 1, :]  # Shape: [num_samples, num_currencies]
        predicted = predictions[:, day - 1, :]  # Shape: [num_samples, num_currencies]

        metrics = compute_metrics(observed, predicted)

        # Create a DataFrame for the metrics
        metrics_df = pd.DataFrame(metrics, index=CURRENCIES)

        # Add "Average" row: mean of each metric across all currencies
        metrics_df.loc["Average"] = metrics_df.mean(axis=0)

        # Save results to CSV
        metrics_csv = os.path.join(RESULTS_DIR, f"evaluation_metrics_day_{day}.csv")
        metrics_df.to_csv(metrics_csv, index=True)
        print(f"Metrics saved for day {day} to {metrics_csv}")

def main():
    try:
        evaluate_model()
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        exit(1)

if __name__ == "__main__":
    main()   