import pandas as pd
import os
import glob

# --- Configuration ---
# These labels should ideally be consistent with the script that generates these files.
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
INPUT_FOLDER = "." # Assuming CSVs are in the same folder as the script
# Output will be an update to the input files themselves.

def populate_labels_in_file(filepath, emotion_labels):
    """
    Reads a CSV file, identifies the highest scored emotion for each row,
    populates the 'manual_emotion_label' column, and saves the file.
    """
    print(f"Processing file: {filepath}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  Read {len(df)} rows from {filepath}")
    except FileNotFoundError:
        print(f"  Error: Input file not found at {filepath}")
        return
    except Exception as e:
        print(f"  Error reading CSV {filepath}: {e}")
        return

    emotion_score_columns = [f'emotion_{label}' for label in emotion_labels]

    # Check if all required emotion score columns exist
    missing_cols = [col for col in emotion_score_columns if col not in df.columns]
    if missing_cols:
        print(f"  Error: Missing expected emotion score columns: {', '.join(missing_cols)}. Skipping file.")
        return

    if 'manual_emotion_label' not in df.columns:
        print("  Error: CSV must contain 'manual_emotion_label' column. Skipping file.")
        # Optionally, create the column if it's missing:
        # df['manual_emotion_label'] = "" 
        # print("  Warning: 'manual_emotion_label' column was missing; it has been created.")
        return

    # Identify the column with the maximum score for each row
    # This will be NaN if all scores in a row are NaN
    df['highest_emotion_col_temp'] = df[emotion_score_columns].idxmax(axis=1)

    # Convert the column name (e.g., 'emotion_joy') to the label (e.g., 'joy')
    # .str.replace will correctly handle NaN values in 'highest_emotion_col_temp'
    df['manual_emotion_label'] = df['highest_emotion_col_temp'].str.replace('emotion_', '', regex=False)

    # Drop the temporary column
    df.drop(columns=['highest_emotion_col_temp'], inplace=True)
    
    # For rows where all emotion scores were NaN, 'manual_emotion_label' will be NaN.
    # Pandas to_csv by default writes NaN as empty strings, which is often desired.
    # If specific handling for all-NaN rows is needed (e.g. "unknown"), it can be added here:
    # df.loc[df[emotion_score_columns].isnull().all(axis=1), 'manual_emotion_label'] = "unknown" # Example

    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"  Successfully updated and saved data to {filepath}")
    except Exception as e:
        print(f"  Error saving updated CSV to {filepath}: {e}")

def main():
    print("Starting script to populate manual emotion labels...")

    # Find files matching the pattern *_emotion_labelling_data.csv
    search_pattern = os.path.join(INPUT_FOLDER, "*_emotion_labelling_data.csv")
    files_to_process = glob.glob(search_pattern)

    if not files_to_process:
        print(f"No files found matching the pattern '{search_pattern}' in '{os.path.abspath(INPUT_FOLDER)}'.")
        print("Please ensure the input files from 'prepare_emotion_labelling_dataset.py' are present.")
        return

    print(f"Found {len(files_to_process)} files to process:")
    for f_path in files_to_process:
        print(f"  - {f_path}")

    for filepath in files_to_process:
        populate_labels_in_file(filepath, EMOTION_LABELS)

    print("\\nScript finished.")

if __name__ == "__main__":
    # Ensure the script's working directory is where the CSVs are,
    # or adjust INPUT_FOLDER accordingly.
    main() 