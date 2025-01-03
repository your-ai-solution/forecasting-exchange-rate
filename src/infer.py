import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    x = []
    for i in range(len(test_df) - sliding_window):
        x.append(test_df.iloc[i:i + sliding_window].values)
    return torch.tensor(np.array(x), dtype=torch.float32)

def predict_and_save_results(model, test_data):
    """Generate predictions and save them as CSV."""
    print("Generating predictions...")
    predictions = []
    with torch.no_grad():
        for i in range(len(test_data)):
            input_data = test_data[i].unsqueeze(0)
            pred = model(input_data).squeeze(0).numpy()
            predictions.append(pred)

    predictions = np.array(predictions)
    for day in range(FORECAST_HORIZON):
        results = pd.DataFrame(predictions[:, day, :], columns=CURRENCIES)
        results.to_csv(os.path.join(RESULTS_DIR, f"predictions_day_{day + 1}.csv"), index=False)
        print(f"Predictions saved for day {day + 1}.")

def plot_results(test_df):
    """Plot observed and predicted time series."""
    print("Generating plots...")

    # Define colors for currencies
    colors = plt.cm.tab10(range(len(CURRENCIES)))  # Use a colormap for distinct colors

    for day in range(1, FORECAST_HORIZON + 1):
        predictions_file = os.path.join(RESULTS_DIR, f"predictions_day_{day}.csv")
        predictions = pd.read_csv(predictions_file)

        plt.figure(figsize=(12, 6))
        for i, currency in enumerate(CURRENCIES):
            # Align observed and predicted data for visualization
            observed = test_df.iloc[SLIDING_WINDOW + day - 1:, i].reset_index(drop=True)  # Trim observed
            predicted = predictions[currency].reset_index(drop=True)

            # Extend predictions to match observed length (if shorter)
            if len(predicted) < len(observed):
                predicted = pd.Series(
                    list(predicted) + [np.nan] * (len(observed) - len(predicted))
                )

            # Plot observed and predicted series with the same color
            plt.plot(observed, label=f"Observed {currency}", linestyle="-", color=colors[i])
            plt.plot(predicted, label=f"Predicted {currency}", linestyle="--", color=colors[i])

        plt.title(f"Observed vs Predicted for Day {day}")
        plt.xlabel("Time")
        plt.ylabel("Exchange Rate")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_day_{day}.png"))
        plt.close()
        print(f"Plot saved for Day {day} forecast.")

def main():
    try:
        # Load test data
        print("Loading test data...")
        test_df = pd.read_csv(TEST_FILE)
        test_data = prepare_test_data(test_df, sliding_window=SLIDING_WINDOW)

        # Load model and configuration
        model_path = os.path.join(CONFIGS_DIR, "transformer_model_7_day.pth")
        config_path = model_path.replace(".pth", "_config.json")
        model = load_model(model_path, config_path)

        # Predict and save results
        predict_and_save_results(model, test_data)

        # Generate and save plots
        plot_results(test_df)

        print("Inference and visualization completed successfully!")
    except Exception as e:
        print(f"Error during inference: {e}")
        exit(1)

if __name__ == "__main__":
    main()