import os
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import torch.nn as nn
import torch.optim as optim

# Paths
PREPROCESSED_DIR = "data/preprocessed"
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, "train.csv")
VALID_FILE = os.path.join(PREPROCESSED_DIR, "valid.csv")
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SLIDING_WINDOW = 30
FORECAST_HORIZON = 7  # Predict for 1 to 7 days ahead
NUM_CURRENCIES = 8  # Number of currencies

# Dataset class for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sliding_window, forecast_horizon):
        self.data = data
        self.sliding_window = sliding_window
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.sliding_window - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.sliding_window, :]
        y = self.data[idx + self.sliding_window : idx + self.sliding_window + self.forecast_horizon, :]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# LightningModule wrapper for transformers
class TransformerForecastingModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, forecast_horizon, num_currencies, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()

        # Currency embedding
        self.currency_embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True
        )

        # Decoder
        self.decoder = nn.Linear(hidden_dim, forecast_horizon * num_currencies)

        # Loss function
        self.criterion = nn.MSELoss()

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, x):
        # Apply currency embedding
        x = self.currency_embedding(x)

        # Transformer
        x = self.transformer(x, x)

        # Decode to output shape: [batch_size, forecast_horizon, num_currencies]
        x = self.decoder(x[:, -1, :])
        return x.view(x.size(0), self.hparams.forecast_horizon, self.hparams.num_currencies)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

def load_data():
    """
    Load train and validation data.
    """
    print("Loading train and validation data...")
    train_data = pd.read_csv(TRAIN_FILE).to_numpy()
    valid_data = pd.read_csv(VALID_FILE).to_numpy()
    return train_data, valid_data

def train_transformer(train_data, valid_data, sliding_window, forecast_horizon, num_currencies, model_path):
    """
    Train a transformer for time series forecasting.
    """
    print(f"Training a transformer for {forecast_horizon}-day forecast...")

    # Prepare datasets
    train_dataset = TimeSeriesDataset(train_data, sliding_window, forecast_horizon)
    valid_dataset = TimeSeriesDataset(valid_data, sliding_window, forecast_horizon)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # Initialize LightningModule
    model = TransformerForecastingModel(
        input_dim=train_data.shape[1],
        hidden_dim=64,
        num_layers=4,
        forecast_horizon=forecast_horizon,
        num_currencies=num_currencies,
    )

    # Train the model
    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10), LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(model, train_loader, valid_loader)

    # Save model and configuration
    torch.save(model.state_dict(), model_path)
    config = {
        "input_dim": train_data.shape[1],
        "hidden_dim": 64,
        "num_layers": 4,
        "forecast_horizon": forecast_horizon,
        "num_currencies": num_currencies,
    }
    with open(model_path.replace(".pth", "_config.json"), "w") as f:
        json.dump(config, f)
    print(f"Model and configuration saved to {model_path} and {model_path.replace('.pth', '_config.json')}")

def main():
    try:
        # Step 1: Load data
        train_data, valid_data = load_data()

        # Step 2: Train model for 7-day forecast
        model_path = os.path.join(CONFIGS_DIR, "transformer_model_7_day.pth")
        train_transformer(train_data, valid_data, SLIDING_WINDOW, FORECAST_HORIZON, NUM_CURRENCIES, model_path)

        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        exit(1)

if __name__ == "__main__":
    main()