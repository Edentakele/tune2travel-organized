import pandas as pd
import numpy as np
import os
import torch
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel, Features, Value
import joblib # For potential future saving/loading

# --- Configuration (Should match hf_spam_classifier.py for data loading) ---
# Use absolute paths derived during previous fixes
SPAM_DATA_DIR = '/run/media/ottobeeth/362A4A2E2A49EC05/Users/ottobeeth/courses/tune2travel/emotion_analysis/spam_detector_2000/dataset_approach'
NON_SPAM_FILE = '/run/media/ottobeeth/362A4A2E2A49EC05/Users/ottobeeth/courses/tune2travel/data/yearly_data/despa_2018_en_nospam_labelled_default.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
HF_MODEL_DIR = './hf_spam_classifier_results' # Directory where the fine-tuned HF model is saved
NUM_NON_SPAM_SAMPLES = 2000 # Consistent non-spam sampling

# --- LR Model Configuration (from spam_detector.py) ---
LR_MAX_ITER = 1000
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = 'english'

# --- 1. Load and Split Data (Replicating hf_spam_classifier.py logic) ---
print("Loading data exactly as in hf_spam_classifier.py...")

# Load Spam Data
spam_files_list = glob.glob(os.path.join(SPAM_DATA_DIR, '*.csv'))
if not spam_files_list:
    raise FileNotFoundError(f"No CSV files found in {SPAM_DATA_DIR}.")

spam_datasets = []
for f in spam_files_list:
    try:
        ds = load_dataset('csv', data_files=f, split='train')
        if 'CONTENT' in ds.column_names and 'CLASS' in ds.column_names:
            ds = ds.rename_columns({'CONTENT': 'text', 'CLASS': 'label'})
            ds = ds.select_columns(['text', 'label'])
            ds = ds.filter(lambda example: example['label'] == 1)
            if len(ds) > 0:
                spam_datasets.append(ds)
    except Exception as e:
        print(f"Error loading spam file {f}: {e}")

if not spam_datasets:
    raise ValueError("No valid spam data could be loaded.")

spam_data_full = concatenate_datasets(spam_datasets)
print(f"Loaded {len(spam_data_full)} spam samples.")

# Load Non-Spam Data
try:
    non_spam_dataset = load_dataset('csv', data_files=NON_SPAM_FILE, split='train')
    if 'comment' not in non_spam_dataset.column_names:
        raise ValueError(f"Missing 'comment' column in {NON_SPAM_FILE}")
    non_spam_dataset = non_spam_dataset.rename_column('comment', 'text')
    non_spam_dataset = non_spam_dataset.select_columns(['text'])
    non_spam_dataset = non_spam_dataset.map(lambda example: {'label': 0})

    # Sample Non-Spam Data
    if len(non_spam_dataset) > NUM_NON_SPAM_SAMPLES:
        print(f"Sampling {NUM_NON_SPAM_SAMPLES} non-spam samples...")
        non_spam_dataset = non_spam_dataset.shuffle(seed=RANDOM_STATE).select(range(NUM_NON_SPAM_SAMPLES))
    else:
         print(f"Using all {len(non_spam_dataset)} available non-spam samples.")
    print(f"Using {len(non_spam_dataset)} non-spam samples.")

except Exception as e:
     raise RuntimeError(f"Error loading or sampling non-spam data: {e}")

# Combine Data
print("Combining and shuffling datasets...")
combined_dataset = concatenate_datasets([spam_data_full, non_spam_dataset])
combined_dataset = combined_dataset.map(lambda example: {'text': str(example['text']) if example['text'] is not None else ''})
combined_dataset = combined_dataset.shuffle(seed=RANDOM_STATE)

# Cast label column
print("Casting label column...")
new_features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=['Non-Spam', 'Spam'])
})
combined_dataset = combined_dataset.cast(new_features)

# Split Data
print("Splitting dataset...")
dataset_split = combined_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_STATE, stratify_by_column='label')
train_dataset = dataset_split['train']
test_dataset = dataset_split['test']

# Extract text and labels for convenience (needed for LR training/evaluation)
X_train_text = [item['text'] for item in train_dataset]
y_train_labels = [item['label'] for item in train_dataset]
X_test_text = [item['text'] for item in test_dataset]
y_test_labels = [item['label'] for item in test_dataset] # True labels for evaluation

print(f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")

# --- 2. Train Logistic Regression Model ---
print("\n--- Training Logistic Regression Model ---")
# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    stop_words=TFIDF_STOP_WORDS,
    max_features=TFIDF_MAX_FEATURES
)

# Create Logistic Regression Classifier
lr_classifier = LogisticRegression(
    max_iter=LR_MAX_ITER,
    random_state=RANDOM_STATE
)

# Create Pipeline
lr_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', lr_classifier)
])

# Train the LR pipeline on the training data
print("Fitting LR pipeline...")
lr_pipeline.fit(X_train_text, y_train_labels)
print("LR pipeline trained.")

# --- 3. Load Fine-tuned Hugging Face Model ---
print("\n--- Loading Fine-tuned Hugging Face Model ---")
if not os.path.exists(HF_MODEL_DIR):
    raise FileNotFoundError(f"Hugging Face model directory not found: {HF_MODEL_DIR}. Did you run hf_spam_classifier.py?")

try:
    print(f"Loading tokenizer from {HF_MODEL_DIR}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    print(f"Loading model from {HF_MODEL_DIR}...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR)
    print("Creating HF pipeline...")
    hf_pipe = TextClassificationPipeline(
        model=hf_model,
        tokenizer=hf_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("HF model and pipeline loaded.")
except Exception as e:
    raise RuntimeError(f"Error loading Hugging Face model/tokenizer from {HF_MODEL_DIR}: {e}")

# --- 4. Predict and Evaluate ---
print("\n--- Generating Predictions ---")

# Predict with Logistic Regression
print("Predicting with LR model...")
lr_predictions = lr_pipeline.predict(X_test_text) # Returns label IDs (0 or 1)

# Predict with Hugging Face Model
print("Predicting with HF model...")
# The HF pipeline expects raw text. It returns a list of dicts.
# Example: [{'label': 'LABEL_1', 'score': 0.99}] or [{'label': 'LABEL_0', 'score': 0.8}]
# We need to extract the predicted label ID.
hf_raw_predictions = hf_pipe(X_test_text, batch_size=16, truncation=True) # Use a reasonable batch size

hf_predictions = []
for pred in hf_raw_predictions:
    # Assuming labels are 'LABEL_0' (Non-Spam) and 'LABEL_1' (Spam)
    label_id = int(pred['label'].split('_')[-1])
    hf_predictions.append(label_id)

print("Predictions generated.")

# --- Evaluation ---
print("\n--- Model Evaluation --- ")

target_names = ['Non-Spam', 'Spam'] # From ClassLabel definition

print("\n--- Logistic Regression Performance ---")
lr_accuracy = accuracy_score(y_test_labels, lr_predictions)
lr_report = classification_report(y_test_labels, lr_predictions, target_names=target_names, zero_division=0)
print(f"Accuracy: {lr_accuracy:.4f}")
print("Classification Report:")
print(lr_report)

print("\n--- Hugging Face Model Performance ---")
hf_accuracy = accuracy_score(y_test_labels, hf_predictions)
hf_report = classification_report(y_test_labels, hf_predictions, target_names=target_names, zero_division=0)
print(f"Accuracy: {hf_accuracy:.4f}")
print("Classification Report:")
print(hf_report)

# --- Summary ---
print("\n--- Performance Summary ---")
print(f"                     | Logistic Regression | Hugging Face (BERT) | ")
print(f"---------------------|---------------------|---------------------|")
print(f"Accuracy             | {lr_accuracy:<19.4f} | {hf_accuracy:<19.4f} |")

# Extract specific metrics for cleaner table (optional)
lr_report_dict = classification_report(y_test_labels, lr_predictions, target_names=target_names, zero_division=0, output_dict=True)
hf_report_dict = classification_report(y_test_labels, hf_predictions, target_names=target_names, zero_division=0, output_dict=True)

print(f"Precision (Spam)   | {lr_report_dict['Spam']['precision']:<19.4f} | {hf_report_dict['Spam']['precision']:<19.4f} |")
print(f"Recall (Spam)      | {lr_report_dict['Spam']['recall']:<19.4f} | {hf_report_dict['Spam']['recall']:<19.4f} |")
print(f"F1-Score (Spam)    | {lr_report_dict['Spam']['f1-score']:<19.4f} | {hf_report_dict['Spam']['f1-score']:<19.4f} |")
print(f"Precision (Non-Spam)| {lr_report_dict['Non-Spam']['precision']:<19.4f} | {hf_report_dict['Non-Spam']['precision']:<19.4f} |")
print(f"Recall (Non-Spam)   | {lr_report_dict['Non-Spam']['recall']:<19.4f} | {hf_report_dict['Non-Spam']['recall']:<19.4f} |")
print(f"F1-Score (Non-Spam)| {lr_report_dict['Non-Spam']['f1-score']:<19.4f} | {hf_report_dict['Non-Spam']['f1-score']:<19.4f} |")

print("\nComparison script finished.") 