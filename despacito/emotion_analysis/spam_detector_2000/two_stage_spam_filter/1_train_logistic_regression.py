import pandas as pd
import numpy as np
import glob
import os
import joblib # Added for saving the model
from sklearn.model_selection import train_test_split # Removed learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Removed MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Removed matplotlib imports

# --- Configuration ---
# Use relative paths from the project root directory
SPAM_DATA_DIR = 'spam_detector_2000/dataset_approach' # Corrected path relative to project root
NON_SPAM_FILE = 'data/topic_csv/cleaned_despacito.csv' # Path relative to project root
N_SAMPLES = 2000
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Output model path relative to project root, saving within the script's folder structure
OUTPUT_MODEL_DIR = 'spam_detector_2000/two_stage_spam_filter'
OUTPUT_MODEL_FILE = os.path.join(OUTPUT_MODEL_DIR, 'logistic_regression_spam_pipeline.joblib')

# Ensure output directory exists before saving model later
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# --- Load Spam Data ---
# ... (Keep existing spam data loading logic) ...
print(f"Loading spam data from: {SPAM_DATA_DIR}")
# Check if the directory exists relative to the current working directory (assumed to be project root)
if not os.path.isdir(SPAM_DATA_DIR):
     raise FileNotFoundError(f"Spam data directory not found at project root level: {SPAM_DATA_DIR}. Please ensure the UCI dataset is extracted correctly in the root.")

spam_files = glob.glob(os.path.join(SPAM_DATA_DIR, '*.csv'))
if not spam_files:
    # Removed fallback logic
    raise FileNotFoundError(f"No CSV files found in {SPAM_DATA_DIR}. Please check the directory contents.")


spam_dfs = []
for f in spam_files:
    try:
        df = pd.read_csv(f)
        # Standardize column names - assuming typical UCI dataset structure
        if 'CONTENT' in df.columns and 'CLASS' in df.columns:
             df = df[['CONTENT', 'CLASS']]
             df.columns = ['text', 'label'] # Rename for consistency
             # Ensure label is integer
             df['label'] = df['label'].astype(int)
             spam_dfs.append(df[df['label'] == 1]) # Keep only spam comments
        else:
             print(f"Warning: Skipping file {f} due to missing 'CONTENT' or 'CLASS' columns.")
    except Exception as e:
        print(f"Error loading file {f}: {e}")

if not spam_dfs:
     raise ValueError("No valid spam data could be loaded. Check CSV files and column names ('CONTENT', 'CLASS').")

spam_data = pd.concat(spam_dfs, ignore_index=True)
# Ensure label is integer type after concat
spam_data['label'] = spam_data['label'].astype(int)
# Sample after ensuring integer type
spam_data = spam_data.sample(n=min(N_SAMPLES, len(spam_data)), random_state=RANDOM_STATE)
print(f"Loaded {len(spam_data)} spam samples.")


# --- Load Non-Spam Data ---
# ... (Keep existing non-spam data loading logic, adjust path if needed) ...
print(f"Loading non-spam data from: {NON_SPAM_FILE}")
try:
    # Check if file exists relative to the current working directory (assumed to be project root)
    if not os.path.exists(NON_SPAM_FILE):
        # Removed fallback logic
        raise FileNotFoundError(f"Non-spam file not found at project root level: {NON_SPAM_FILE}")

    non_spam_data = pd.read_csv(NON_SPAM_FILE)
    # Check required column
    if 'processed_comment' not in non_spam_data.columns:
        raise ValueError(f"Missing 'processed_comment' column in {NON_SPAM_FILE}")

    non_spam_data = non_spam_data[['processed_comment']].copy()
    non_spam_data.columns = ['text'] # Rename for consistency
    non_spam_data['label'] = 0 # Assign non-spam label
    non_spam_data = non_spam_data.sample(n=min(N_SAMPLES, len(non_spam_data)), random_state=RANDOM_STATE)
    print(f"Loaded {len(non_spam_data)} non-spam samples.")
except FileNotFoundError as e:
     print(f"Error: {e}") # Print specific error
     raise e
except Exception as e:
     raise RuntimeError(f"Error loading non-spam data: {e}")


# --- Combine Data ---
# ... (Keep existing data combination logic) ...
print("Combining spam and non-spam data...")
combined_data = pd.concat([spam_data, non_spam_data], ignore_index=True)
# Ensure label is integer after combining
combined_data['label'] = combined_data['label'].astype(int)
# Handle potential NaN values in text column resulting from loading issues or empty comments
combined_data['text'] = combined_data['text'].fillna('')
print(f"Total samples: {len(combined_data)}")
print(f"Class distribution:\n{combined_data['label'].value_counts()}")

# --- Data Splitting ---
# ... (Keep existing data splitting logic) ...
print("Splitting data into training and testing sets...")
X = combined_data['text']
y = combined_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# --- Model Training and Evaluation (Logistic Regression Only) ---
classifier_name = "Logistic Regression"
classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

print(f"\n--- Training and Evaluating {classifier_name} ---")
# Create pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', classifier)
])

# Train
print(f"Training {classifier_name}...")
pipeline.fit(X_train, y_train)

# Evaluate (Optional but good practice)
print(f"Evaluating {classifier_name} on test set...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam'])

print(f"\n{classifier_name} Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# --- Save the Trained Pipeline ---
print(f"\nSaving the trained {classifier_name} pipeline to {OUTPUT_MODEL_FILE}...")
# The directory OUTPUT_MODEL_DIR was created earlier
joblib.dump(pipeline, OUTPUT_MODEL_FILE)
print(f"Pipeline saved successfully to {OUTPUT_MODEL_FILE}")

# --- Remove Learning Curve Plotting ---
# ... (Section removed) ...

print("\nScript finished.") 