import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# --- Configuration ---

# Spam Detection Configuration (adapted from spam_detector.py)
SPAM_DATA_DIR = './emotion_analysis/spam_detector_2000/dataset_approach'  # Path to UCI spam dataset
NON_SPAM_FILE = './data/topic_csv/cleaned_despacito.csv' # Path to your non-spam data
N_SAMPLES_SPAM_TRAIN = 2000  # Number of samples for training spam detector
RANDOM_STATE = 42

# Emotion Detection Configuration
EMOTION_MODEL_DIR = 'emotion_analysis/emotion/manually_labelled/emotion_model_output_best_hyper' # Relative to this script's parent dir
EMOTION_BATCH_SIZE = 32
# id2label for emotion model (extracted from its config.json)
EMOTION_ID2LABEL = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

# Input/Output Configuration
INPUT_CSV_FILES = [
    './final_extraction/despa_kJQP7kiw5Fk_comments.csv',
    './final_extraction/seeyou_RgKAFK5djSk_comments.csv'
]
COMMENT_COLUMN_NAME = 'original_comment'  # CHANGE THIS if your comment column has a different name
OUTPUT_SPAM_SUFFIX = '_spam_labelled.csv'
OUTPUT_EMOTION_SUFFIX = '_spam_emotion_labelled.csv'

# --- Helper Functions ---
def preprocess_text_for_emotion(text):
    if not isinstance(text, str):
        return ""
    # Basic cleaning: remove URLs, user mentions, and non-alphanumeric characters (except spaces)
    text = re.sub(r'http\S+|www\.\S+', '', text) # remove URLs
    text = re.sub(r'@\w+', '', text) # remove user mentions
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # remove special characters
    text = text.lower() # convert to lowercase
    text = text.strip() # remove leading/trailing whitespace
    return text

# --- 1. Train Spam Detector ---
print("--- Training Spam Detector ---")

# Load Spam Data
print(f"Loading spam data for training from: {SPAM_DATA_DIR}")
spam_files_train = glob.glob(os.path.join(SPAM_DATA_DIR, '*.csv'))
if not spam_files_train:
    raise FileNotFoundError(f"No CSV files found in {SPAM_DATA_DIR} for spam training.")

spam_dfs_train = []
for f_train in spam_files_train:
    try:
        df_train = pd.read_csv(f_train)
        if 'CONTENT' in df_train.columns and 'CLASS' in df_train.columns:
            df_train = df_train[['CONTENT', 'CLASS']]
            df_train.columns = ['text', 'label']
            spam_dfs_train.append(df_train[df_train['label'] == 1])
        else:
            print(f"Warning: Skipping training file {f_train} due to missing 'CONTENT' or 'CLASS' columns.")
    except Exception as e:
        print(f"Error loading training file {f_train}: {e}")

if not spam_dfs_train:
    raise ValueError("No valid spam data could be loaded for training.")

spam_data_train = pd.concat(spam_dfs_train, ignore_index=True)
spam_data_train = spam_data_train.sample(n=min(N_SAMPLES_SPAM_TRAIN, len(spam_data_train)), random_state=RANDOM_STATE)
print(f"Loaded {len(spam_data_train)} spam samples for training.")

# Load Non-Spam Data for training
print(f"Loading non-spam data for training from: {NON_SPAM_FILE}")
try:
    non_spam_data_train = pd.read_csv(NON_SPAM_FILE)
    if 'processed_comment' not in non_spam_data_train.columns:
        raise ValueError(f"Missing 'processed_comment' column in {NON_SPAM_FILE}")
    non_spam_data_train = non_spam_data_train[['processed_comment']].copy()
    non_spam_data_train.columns = ['text']
    non_spam_data_train['label'] = 0
    non_spam_data_train = non_spam_data_train.sample(n=min(N_SAMPLES_SPAM_TRAIN, len(non_spam_data_train)), random_state=RANDOM_STATE)
    print(f"Loaded {len(non_spam_data_train)} non-spam samples for training.")
except FileNotFoundError:
    raise FileNotFoundError(f"Non-spam training file not found: {NON_SPAM_FILE}")
except Exception as e:
    raise RuntimeError(f"Error loading non-spam training data: {e}")

# Combine and Prepare Training Data
combined_data_train = pd.concat([spam_data_train, non_spam_data_train], ignore_index=True)
combined_data_train['text'] = combined_data_train['text'].fillna('')
X_train_spam = combined_data_train['text']
y_train_spam = combined_data_train['label']

# Create and Train Spam Pipeline (Logistic Regression)
print("Training Logistic Regression spam model...")
spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
spam_pipeline.fit(X_train_spam, y_train_spam)
print("Spam detector trained successfully.")

# --- 2. Load Emotion Detection Model ---
print("\n--- Loading Emotion Detection Model ---")
emotion_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Adjust path to be relative to the script's location if necessary
emotion_model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EMOTION_MODEL_DIR)

if not os.path.exists(emotion_model_full_path):
    # Try path relative to workspace root if the above fails (common in some execution environments)
    # This assumes the script is run from workspace_root/emotion_analysis/
    emotion_model_full_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', EMOTION_MODEL_DIR.replace('./','')))


if not os.path.exists(emotion_model_full_path):
     print(f"Error: Emotion model directory not found at {emotion_model_full_path} or fallback path.")
     print("Please check the EMOTION_MODEL_DIR path.")
     # As a last resort, try the path assuming script is run from workspace root
     emotion_model_full_path = os.path.abspath(EMOTION_MODEL_DIR)
     if not os.path.exists(emotion_model_full_path):
        raise FileNotFoundError(f"Emotion model directory not found. Tried: {emotion_model_full_path} and variations.")


print(f"Loading emotion model from: {emotion_model_full_path}")
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_full_path)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_full_path).to(emotion_device)
emotion_model.eval()
print("Emotion detection model loaded successfully.")


# --- 3. Process Input CSV Files ---
print("\n--- Processing Input CSV Files ---")

for input_csv_path_rel in INPUT_CSV_FILES:
    # Construct absolute path for input CSV
    input_csv_path_abs = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), input_csv_path_rel))
    
    if not os.path.exists(input_csv_path_abs):
        print(f"Input file not found: {input_csv_path_abs}. Trying relative to workspace root...")
        # This assumes the script is in a subdir of workspace and INPUT_CSV_FILES are relative to workspace root
        input_csv_path_abs = os.path.abspath(input_csv_path_rel.replace('../../', '')) # Adjust based on your structure
        if not os.path.exists(input_csv_path_abs):
            print(f"Still cannot find {input_csv_path_rel}. Skipping.")
            continue
            
    print(f"\nProcessing file: {input_csv_path_abs}")
    
    try:
        df_input = pd.read_csv(input_csv_path_abs)
    except FileNotFoundError:
        print(f"Error: File not found {input_csv_path_abs}. Skipping.")
        continue
    except Exception as e:
        print(f"Error reading {input_csv_path_abs}: {e}. Skipping.")
        continue

    if COMMENT_COLUMN_NAME not in df_input.columns:
        print(f"Error: Comment column '{COMMENT_COLUMN_NAME}' not found in {input_csv_path_abs}. Skipping.")
        continue

    df_input_cleaned = df_input.dropna(subset=[COMMENT_COLUMN_NAME])
    df_input_cleaned[COMMENT_COLUMN_NAME] = df_input_cleaned[COMMENT_COLUMN_NAME].astype(str)
    comments_to_label = df_input_cleaned[COMMENT_COLUMN_NAME].tolist()

    if not comments_to_label:
        print(f"No comments found in {input_csv_path_abs} after cleaning. Skipping.")
        continue

    # --- 3a. Spam Labeling ---
    print("Performing spam labeling...")
    spam_predictions_numeric = spam_pipeline.predict(comments_to_label)
    spam_labels = ['spam' if pred == 1 else 'not_spam' for pred in spam_predictions_numeric]
    
    df_spam_labelled = df_input_cleaned.copy()
    df_spam_labelled['spam_label'] = spam_labels
    
    base, ext = os.path.splitext(input_csv_path_abs)
    output_spam_csv_path = base + OUTPUT_SPAM_SUFFIX
    df_spam_labelled.to_csv(output_spam_csv_path, index=False)
    print(f"Saved spam labeled data to: {output_spam_csv_path}")

    # --- 3b. Emotion Labeling (on original comments, but can be changed to spam-filtered if needed) ---
    print("Performing emotion labeling...")
    
    # Preprocess text for emotion model
    texts_for_emotion = [preprocess_text_for_emotion(text) for text in comments_to_label]

    emotion_preds_numeric = []
    with torch.no_grad():
        for i in range(0, len(texts_for_emotion), EMOTION_BATCH_SIZE):
            batch_texts = texts_for_emotion[i:i+EMOTION_BATCH_SIZE]
            if not any(batch_texts): # Skip if batch is all empty strings after preprocessing
                emotion_preds_numeric.extend([-1] * len(batch_texts)) # Use -1 or some placeholder for empty/invalid
                continue
            enc = emotion_tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
            for k_enc in enc:
                enc[k_enc] = enc[k_enc].to(emotion_device)
            outputs = emotion_model(**enc)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            emotion_preds_numeric.extend(batch_preds)
            
    emotion_labels = [EMOTION_ID2LABEL.get(pred, "unknown") for pred in emotion_preds_numeric]

    df_emotion_labelled = df_spam_labelled.copy() # Start with the spam-labelled dataframe
    df_emotion_labelled['emotion_label'] = emotion_labels
    
    output_emotion_csv_path = base + OUTPUT_EMOTION_SUFFIX
    df_emotion_labelled.to_csv(output_emotion_csv_path, index=False)
    print(f"Saved spam and emotion labeled data to: {output_emotion_csv_path}")

print("\n--- Script Finished ---") 