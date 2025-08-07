import pandas as pd
import joblib
import os
import argparse

# --- Configuration ---
# Path to the saved Logistic Regression pipeline (relative to project root)
LR_MODEL_PATH = 'spam_detector_2000/two_stage_spam_filter/logistic_regression_spam_pipeline.joblib'
INTERNAL_TEXT_COLUMN_NAME = 'comment' # Define a standard internal name

# --- Function Definitions ---
def load_data(filepath, text_column_index):
    """Loads data from a CSV file without headers, using column index."""
    print(f"Loading data from: {filepath} (assuming no header)")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    try:
        # Read CSV without header
        df = pd.read_csv(filepath, header=None)

        # Validate text_column_index
        if not isinstance(text_column_index, int) or text_column_index < 0 or text_column_index >= len(df.columns):
             raise ValueError(f"Invalid text column index '{text_column_index}'. File has {len(df.columns)} columns (0-based index). Available indices: {list(df.columns)}")

        # Keep only the specified text column initially, handle NaN
        df = df[[text_column_index]].copy()
        df.rename(columns={text_column_index: INTERNAL_TEXT_COLUMN_NAME}, inplace=True) # Rename to standard internal name
        df[INTERNAL_TEXT_COLUMN_NAME] = df[INTERNAL_TEXT_COLUMN_NAME].fillna('')
        print(f"Loaded {len(df)} records using column index {text_column_index} as text.")
        return df
    except ValueError as e:
        raise e # Re-raise ValueError for specific index errors
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}")

def filter_spam(df, model_path):
    """Filters spam using the pre-trained Logistic Regression model."""
    print(f"Loading Logistic Regression model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LR model file not found: {model_path}")
    try:
        lr_pipeline = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading LR model: {e}")

    print(f"Predicting spam/non-spam using Logistic Regression on '{INTERNAL_TEXT_COLUMN_NAME}' column...")
    # Use the standard internal column name for prediction
    predictions = lr_pipeline.predict(df[INTERNAL_TEXT_COLUMN_NAME])
    df['lr_spam_prediction'] = predictions

    non_spam_df = df[df['lr_spam_prediction'] == 0].copy()
    num_spam = len(df) - len(non_spam_df)
    print(f"Filtered out {num_spam} comments predicted as spam by LR.")
    print(f"{len(non_spam_df)} comments remaining for second stage analysis.")
    return df

def save_results(df, output_path):
    """Saves the final dataframe with predictions."""
    print(f"Saving results to: {output_path}")
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Keep only the original text and LR prediction
        columns_to_save = [INTERNAL_TEXT_COLUMN_NAME, 'lr_spam_prediction']
        df_to_save = df[columns_to_save].copy()

        # Rename columns for clarity in the output file
        df_to_save.rename(columns={
            INTERNAL_TEXT_COLUMN_NAME: 'original_text', # Rename original text column
            'lr_spam_prediction': 'is_spam_prediction_lr'
        }, inplace=True)

        df_to_save.to_csv(output_path, index=False)
        print("Results saved successfully.")
    except Exception as e:
        raise RuntimeError(f"Error saving results to {output_path}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam filtering using a Logistic Regression model on CSV files (potentially without headers).")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input CSV file containing comments.")
    parser.add_argument("-t", "--text_column_index", required=True, type=int, help="Zero-based index of the column containing the text comments in the input file.")
    parser.add_argument("-o", "--output_file", required=True, help="Path to save the output CSV file with predictions ('original_text', 'is_spam_prediction_lr').")
    parser.add_argument("--lr_model", default=LR_MODEL_PATH, help=f"Path to the trained Logistic Regression pipeline (default: {LR_MODEL_PATH})")

    args = parser.parse_args()

    try:
        # 1. Load Data using column index
        input_df = load_data(args.input_file, args.text_column_index)

        # 2. Filter Spam using Logistic Regression (pass df and model path)
        df_with_predictions = filter_spam(input_df, args.lr_model)

        # 3. Save Results
        save_results(df_with_predictions, args.output_file)

        print("\nLogistic Regression spam filtering finished successfully!")

    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        print(f"\nError during processing: {e}")
        # Consider adding more specific error handling or logging
        exit(1) # Exit with a non-zero code indicates an error 