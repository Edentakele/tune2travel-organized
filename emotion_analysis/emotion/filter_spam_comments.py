import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
# Paths are relative to the workspace root (C:/Users/ottobeeth/courses/tune2travel)
SPAM_DATA_DIR = "emotion_analysis/spam_detector_2000/dataset_approach"
NON_SPAM_FILE = "data/topic_csv/cleaned_despacito.csv"

# Target files to clean (ensure these paths are correct relative to workspace root)
FILES_TO_CLEAN = [
    "final_extraction/despa_kJQP7kiw5Fk_comments.csv",
    "final_extraction/seeyou_RgKAFK5djSk_comments.csv"
]
# Output directory for cleaned files (relative to workspace root)
OUTPUT_DIR_FOR_CLEANED = "final_extraction"


N_SAMPLES_PER_CLASS = 2000  # Number of samples to try and load for spam and non-spam for training
RANDOM_STATE = 42

def load_training_data(spam_dir, non_spam_file_path, n_samples, random_state):
    """Loads spam and non-spam training data."""
    print(f"Loading spam data from: {spam_dir}")
    spam_files = glob.glob(os.path.join(spam_dir, '*.csv'))
    if not spam_files:
        raise FileNotFoundError(f"No training spam CSV files found in {spam_dir}.")

    spam_dfs = []
    for f in spam_files:
        try:
            df = pd.read_csv(f)
            if 'CONTENT' in df.columns and 'CLASS' in df.columns:
                df_spam_only = df[df['CLASS'] == 1].copy()
                df_spam_only.rename(columns={'CONTENT': 'text', 'CLASS': 'label'}, inplace=True)
                spam_dfs.append(df_spam_only[['text', 'label']])
            else:
                print(f"Warning: Skipping training spam file {f} due to missing 'CONTENT' or 'CLASS' columns.")
        except Exception as e:
            print(f"Error loading training spam file {f}: {e}")

    if not spam_dfs:
        raise ValueError("No valid spam training data could be loaded.")

    spam_data = pd.concat(spam_dfs, ignore_index=True)
    spam_data = spam_data.sample(n=min(n_samples, len(spam_data)), random_state=random_state)
    print(f"Loaded {len(spam_data)} spam samples for training.")

    print(f"Loading non-spam data from: {non_spam_file_path}")
    try:
        non_spam_data = pd.read_csv(non_spam_file_path)
        if 'processed_comment' not in non_spam_data.columns:
            raise ValueError(f"Missing 'processed_comment' column in non-spam training file {non_spam_file_path}")

        non_spam_df = non_spam_data[['processed_comment']].copy()
        non_spam_df.rename(columns={'processed_comment': 'text'}, inplace=True)
        non_spam_df['label'] = 0  # 0 for non-spam
        non_spam_df = non_spam_df.sample(n=min(n_samples, len(non_spam_df)), random_state=random_state)
        print(f"Loaded {len(non_spam_df)} non-spam samples for training.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Non-spam training file not found: {non_spam_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading non-spam training data: {e}")

    combined_data = pd.concat([spam_data, non_spam_df], ignore_index=True)
    combined_data['text'] = combined_data['text'].fillna('') # Handle any NaN text
    print(f"Total training samples: {len(combined_data)}")
    print(f"Training class distribution:\n{combined_data['label'].value_counts(normalize=True)}")
    
    return combined_data['text'], combined_data['label']

def train_spam_classifier(X_train, y_train, random_state):
    """Trains a spam classification pipeline."""
    print("\nTraining spam classifier...")
    # Using Logistic Regression as it's generally good for text classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear'))
    ])
    pipeline.fit(X_train, y_train)
    print("Spam classifier trained successfully.")
    return pipeline

def clean_comment_file(target_file_path, output_file_path, model_pipeline):
    """Loads a comment file, filters out spam from English comments, and saves the cleaned file."""
    print(f"\nProcessing file for cleaning: {target_file_path}")
    try:
        df = pd.read_csv(target_file_path, low_memory=False)
        print(f"  Read {len(df)} comments from {target_file_path}")
    except FileNotFoundError:
        print(f"  Error: File not found at {target_file_path}")
        return
    except Exception as e:
        print(f"  Error reading CSV {target_file_path}: {e}")
        return

    if 'original_comment' not in df.columns or 'detected_language' not in df.columns:
        print(f"  Error: Target CSV {target_file_path} must contain 'original_comment' and 'detected_language' columns.")
        # If 'detected_language' is missing, we can't filter by language for spam detection.
        # Depending on requirements, could proceed to classify all, or skip, or add lang detection.
        # For now, strict requirement: both columns must be present.
        return

    # Separate English and non-English comments
    df_english = df[df['detected_language'].fillna('unknown') == 'en'].copy()
    df_non_english = df[df['detected_language'].fillna('unknown') != 'en'].copy()
    
    print(f"  Found {len(df_english)} English comments and {len(df_non_english)} non-English comments.")

    df_english_non_spam = pd.DataFrame()

    if not df_english.empty:
        # Prepare text for prediction for English comments, handling potential NaNs
        english_texts_to_predict = df_english['original_comment'].fillna('')
        
        print(f"  Predicting spam for {len(english_texts_to_predict)} English comments...")
        english_predictions = model_pipeline.predict(english_texts_to_predict) # 0 for non-spam, 1 for spam
        
        df_english['is_spam_prediction'] = english_predictions
        num_english_spam = df_english['is_spam_prediction'].sum()
        print(f"  Identified {num_english_spam} spam comments out of {len(df_english)} English comments ({num_english_spam/len(df_english)*100 if len(df_english) > 0 else 0:.2f}%).")

        # Filter out spam from English comments
        df_english_non_spam = df_english[df_english['is_spam_prediction'] == 0].copy()
        # Remove the temporary prediction column
        df_english_non_spam.drop(columns=['is_spam_prediction'], inplace=True)
    else:
        print("  No English comments to perform spam detection on.")
        # df_english_non_spam remains an empty DataFrame

    # Combine non-spam English comments with all non-English comments
    # Non-English comments are assumed to be non-spam for this process
    df_cleaned_output = pd.concat([df_english_non_spam, df_non_english], ignore_index=True)
    
    print(f"  Number of comments to save (non-spam English + all non-English): {len(df_cleaned_output)}")

    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df_cleaned_output.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"  Successfully saved language-filtered non-spam data to {output_file_path}")
    except Exception as e:
        print(f"  Error saving output CSV to {output_file_path}: {e}")

def main():
    print("Starting spam filtering script...")

    # Construct absolute paths from workspace root (assuming script is run from workspace root)
    # If script is in final_extraction, and run from there, paths need adjustment (e.g., ../)
    # For simplicity, this script assumes it's run from C:/Users/ottobeeth/courses/tune2travel
    
    # Correctly join paths assuming the script is run from the workspace root
    # If SPAM_DATA_DIR, NON_SPAM_FILE, etc., are already relative to workspace root,
    # no further os.path.join with a workspace root variable is needed here if cwd is workspace root.

    X_train_text, y_train_labels = load_training_data(
        SPAM_DATA_DIR, 
        NON_SPAM_FILE, 
        N_SAMPLES_PER_CLASS, 
        RANDOM_STATE
    )

    # Simple train-test split on the loaded training data just to evaluate the trained model
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X_train_text, y_train_labels, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_labels
    )
    
    spam_model_pipeline = train_spam_classifier(X_train_eval, y_train_eval, RANDOM_STATE) # Train on the 80% portion

    # Evaluate the trained model on the 20% holdout from training data
    print("\nEvaluating spam classifier on holdout training data...")
    y_pred_eval = spam_model_pipeline.predict(X_test_eval)
    accuracy = accuracy_score(y_test_eval, y_pred_eval)
    report = classification_report(y_test_eval, y_pred_eval, target_names=['Non-Spam (0)', 'Spam (1)'])
    print(f"Accuracy on training data holdout: {accuracy:.4f}")
    print("Classification Report (training data holdout):")
    print(report)

    # Now, clean the target files using the model trained on X_train_eval, y_train_eval
    for target_file_relative_path in FILES_TO_CLEAN:
        base_name = os.path.basename(target_file_relative_path)
        name_part, ext_part = os.path.splitext(base_name)
        output_file_name = f"{name_part}_no_spam{ext_part}"
        
        # output_file_path will be in OUTPUT_DIR_FOR_CLEANED
        # target_file_relative_path is like "final_extraction/file.csv"
        # OUTPUT_DIR_FOR_CLEANED is "final_extraction"
        # So, the output path should be os.path.join(OUTPUT_DIR_FOR_CLEANED, output_file_name)
        
        # Assuming script is run from workspace root, target_file_relative_path is correct.
        # And os.path.join(OUTPUT_DIR_FOR_CLEANED, output_file_name) is also correct.
        output_file_full_path = os.path.join(OUTPUT_DIR_FOR_CLEANED, output_file_name)

        clean_comment_file(target_file_relative_path, output_file_full_path, spam_model_pipeline)

    print("\nScript finished.")

if __name__ == "__main__":
    main() 