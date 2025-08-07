import pandas as pd
import random
from transformers import pipeline
import os

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
# The model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SAMPLE_SIZE = 500
INPUT_FOLDER = "." # Assuming CSVs are in the same folder as the script
OUTPUT_FOLDER = "." # Save output in the same folder

def initialize_emotion_pipeline(model_name):
    """Initializes and returns the Hugging Face emotion classification pipeline."""
    print(f"Loading emotion model: {model_name}...")
    try:
        # Using return_all_scores=True to get scores for all labels
        emotion_classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, return_all_scores=True)
        print("Emotion model loaded successfully.")
        return emotion_classifier
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        print("Please ensure you have an internet connection and the transformers library is installed correctly.")
        print("You might need to install it via pip: pip install transformers torch pandas")
        # Depending on your PyTorch setup, you might need a specific torch version.
        # Visit pytorch.org for installation instructions if 'torch' is missing.
        return None

def analyze_emotions(text, classifier):
    """
    Analyzes emotions in a given text using the classifier.
    Returns a dictionary with emotion labels as keys and scores as values.
    """
    if not text or not isinstance(text, str):
        return {label: 0.0 for label in EMOTION_LABELS} # Return zero scores for empty/invalid input

    try:
        results = classifier(text)
        # The pipeline with return_all_scores=True returns a list of lists of dicts
        # e.g., [[{'label': 'sadness', 'score': 0.9...}, {'label': 'joy', 'score': 0.0...}]]
        # We need to flatten it if it's nested and then create a dictionary
        emotion_scores = {}
        if results and isinstance(results, list) and isinstance(results[0], list):
            for emotion_result in results[0]:
                emotion_scores[emotion_result['label']] = round(emotion_result['score'], 4)
        else: # Fallback for unexpected structure, though current models usually return the nested list
             for emotion_result in results:
                emotion_scores[emotion_result['label']] = round(emotion_result['score'], 4)

        # Ensure all desired EMOTION_LABELS are present, add with 0.0 if not
        final_scores = {label: emotion_scores.get(label, 0.0) for label in EMOTION_LABELS}
        return final_scores
    except Exception as e:
        print(f"Error during emotion analysis for text: '{text[:50]}...': {e}")
        return {label: 0.0 for label in EMOTION_LABELS} # Return zero scores on error

def process_comment_file(input_csv_path, output_csv_path, emotion_classifier):
    """
    Reads a CSV (expected to be a _no_spam.csv file), filters for English comments,
    samples up to SAMPLE_SIZE, analyzes emotions, and saves to a new CSV,
    preserving all original columns of the sampled comments.
    """
    print(f"\nProcessing file: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path, low_memory=False)
        print(f"  Read {len(df)} comments from {input_csv_path}")
    except FileNotFoundError:
        print(f"  Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"  Error reading CSV {input_csv_path}: {e}")
        return

    if 'original_comment' not in df.columns or 'detected_language' not in df.columns:
        print("  Error: CSV must contain 'original_comment' and 'detected_language' columns.")
        return

    df_english = df[df['detected_language'].fillna('unknown') == 'en'].copy()
    print(f"  Found {len(df_english)} English comments.")

    if len(df_english) == 0:
        print("  No English comments found. Skipping emotion analysis for this file.")
        # Create an empty DataFrame with expected schema for labelling output
        # Including all columns from the original df if possible, plus new ones
        expected_cols = list(df.columns) if not df.empty else ['comment_id', 'original_comment']
        for label in EMOTION_LABELS:
            expected_cols.append(f'emotion_{label}')
        expected_cols.append('manual_emotion_label')
        empty_df = pd.DataFrame(columns=expected_cols)
        empty_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"  Saved empty dataset with headers to {output_csv_path}")
        return

    if len(df_english) > SAMPLE_SIZE:
        sampled_df = df_english.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"  Sampled {SAMPLE_SIZE} English comments.")
    else:
        sampled_df = df_english
        print(f"  Using all {len(df_english)} English comments (less than or equal to {SAMPLE_SIZE}).")

    output_df = sampled_df.copy() # This preserves all original columns from the sample.

    # Initialize new emotion columns in the output_df
    for label in EMOTION_LABELS:
        output_df[f'emotion_{label}'] = pd.NA

    print(f"  Performing emotion analysis for {len(output_df)} comments...")
    for index, row in output_df.iterrows():
        comment_text = str(row['original_comment'])
        emotions = analyze_emotions(comment_text, emotion_classifier) 
        
        for label in EMOTION_LABELS:
            output_df.loc[index, f'emotion_{label}'] = emotions.get(label, pd.NA)
        
        if (index + 1) % 50 == 0:
            print(f"    Analyzed {index + 1}/{len(output_df)} comments...")

    output_df['manual_emotion_label'] = ""

    try:
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"  Successfully saved processed data with all original columns to {output_csv_path}")
    except Exception as e:
        print(f"  Error saving output CSV to {output_csv_path}: {e}")

def main():
    print("Starting script to prepare emotion labelling dataset...")
    emotion_classifier = initialize_emotion_pipeline(MODEL_NAME)

    if emotion_classifier is None:
        print("Failed to initialize emotion model. Exiting.")
        return

    files_to_process = [
        "despa_kJQP7kiw5Fk_comments_no_spam.csv",
        "seeyou_RgKAFK5djSk_comments_no_spam.csv"
    ]

    for file_name in files_to_process:
        input_path = os.path.join(INPUT_FOLDER, file_name)
        
        base_for_output = file_name
        if file_name.endswith("_comments_no_spam.csv"):
            base_for_output = file_name.replace("_comments_no_spam.csv", "")
        elif file_name.endswith("_comments.csv"): # Fallback for different naming
            base_for_output = file_name.replace("_comments.csv", "")
        else: # More generic fallback, removes extension
            base_for_output = os.path.splitext(file_name)[0]
            # Potentially further strip _no_spam if it exists and wasn't caught by the first case
            if base_for_output.endswith("_no_spam"):
                 base_for_output = base_for_output[:-len("_no_spam")]

        output_file_name = f"{base_for_output}_emotion_labelling_data.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_file_name)
        
        print(f"Input: {input_path}, Output: {output_path}") # Debug print for paths
        process_comment_file(input_path, output_path, emotion_classifier)

    print("\nScript finished.")

if __name__ == "__main__":
    # Make sure the script working directory is where the CSVs are, or adjust INPUT_FOLDER
    # For example, if script is in 'final_extraction' and CSVs too, INPUT_FOLDER = "." is fine.
    # If script is elsewhere, provide full/relative path to 'final_extraction' for INPUT_FOLDER.
    main() 