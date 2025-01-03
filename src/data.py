import os
import gzip
import shutil
import pandas as pd

# Paths
RAW_DATA_FILE = "data/raw/exchange_rate.txt.gz"
PREPROCESSED_DIR = "data/preprocessed"
OUTPUT_FILE = os.path.join(PREPROCESSED_DIR, "exchange_rate.txt")
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, "train.csv")
VALID_FILE = os.path.join(PREPROCESSED_DIR, "valid.csv")
TEST_FILE = os.path.join(PREPROCESSED_DIR, "test.csv")

SLIDING_WINDOW = 30
FORECAST_HORIZONS = [1, 3, 7]

def unzip_gz_file(gz_path, output_path):
    """
    Unzips a .gz file into the specified output file path.
    """
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Compressed file not found: {gz_path}")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    print(f"Unzipping {gz_path} to {output_path}...")
    with gzip.open(gz_path, 'rb') as gz_file:
        with open(output_path, 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)
    print(f"File unzipped successfully to {output_path}.")

def split_and_save_data(txt_file):
    """
    Splits the data into train, validation, and test sets and saves them as CSV files.
    """
    print("Loading and splitting data...")
    
    # Load the TXT file assuming comma-separated values
    data = pd.read_csv(txt_file, delimiter=",", header=None)
    data.columns = ["AUD", "GBP", "CAD", "CHF", "CNY", "JPY", "NZD", "SGD"]

    # Split into train (70%), validation (20%), and test (10%)
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)

    train_df = data[:train_size]
    valid_df = data[train_size:train_size + val_size]
    test_df = data[train_size + val_size:]

    # Save the splits to CSV files
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    valid_df.to_csv(VALID_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"Data split completed.")
    print(f"Train data saved to {TRAIN_FILE}")
    print(f"Validation data saved to {VALID_FILE}")
    print(f"Test data saved to {TEST_FILE}")

def main():
    """
    Main function to handle the preprocessing and splitting process.
    """
    try:
        # Step 1: Unzip the .gz file
        unzip_gz_file(RAW_DATA_FILE, OUTPUT_FILE)

        # Step 2: Split the data and save
        split_and_save_data(OUTPUT_FILE)

        print(f"Dataset preprocessing completed. Preprocessed data is available at: {PREPROCESSED_DIR}")
    except Exception as e:
        print(f"Error during dataset preprocessing: {e}")
        exit(1)

if __name__ == "__main__":
    main()