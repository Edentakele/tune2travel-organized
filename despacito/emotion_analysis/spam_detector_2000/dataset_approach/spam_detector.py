import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Configuration ---
SPAM_DATA_DIR = '/home/ottobeeth/tune2travel/youtube+spam+collection'
NON_SPAM_FILE = '/home/ottobeeth/tune2travel/data/topic_csv/cleaned_despacito.csv'
N_SAMPLES = 2000
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_PLOT_FILE = 'learning_curves.png'

# --- Load Spam Data ---
print(f"Loading spam data from: {SPAM_DATA_DIR}")
spam_files = glob.glob(os.path.join(SPAM_DATA_DIR, '*.csv'))
if not spam_files:
    raise FileNotFoundError(f"No CSV files found in {SPAM_DATA_DIR}. Please ensure the UCI dataset is extracted correctly.")

spam_dfs = []
for f in spam_files:
    try:
        df = pd.read_csv(f)
        # Standardize column names - assuming typical UCI dataset structure
        if 'CONTENT' in df.columns and 'CLASS' in df.columns:
             df = df[['CONTENT', 'CLASS']]
             df.columns = ['text', 'label'] # Rename for consistency
             spam_dfs.append(df[df['label'] == 1]) # Keep only spam comments
        else:
             print(f"Warning: Skipping file {f} due to missing 'CONTENT' or 'CLASS' columns.")
    except Exception as e:
        print(f"Error loading file {f}: {e}")

if not spam_dfs:
     raise ValueError("No valid spam data could be loaded. Check CSV files and column names ('CONTENT', 'CLASS').")

spam_data = pd.concat(spam_dfs, ignore_index=True)
spam_data = spam_data.sample(n=min(N_SAMPLES, len(spam_data)), random_state=RANDOM_STATE)
print(f"Loaded {len(spam_data)} spam samples.")

# --- Load Non-Spam Data ---
print(f"Loading non-spam data from: {NON_SPAM_FILE}")
try:
    non_spam_data = pd.read_csv(NON_SPAM_FILE)
    # Check required column
    if 'processed_comment' not in non_spam_data.columns:
        raise ValueError(f"Missing 'processed_comment' column in {NON_SPAM_FILE}")

    non_spam_data = non_spam_data[['processed_comment']].copy()
    non_spam_data.columns = ['text'] # Rename for consistency
    non_spam_data['label'] = 0 # Assign non-spam label
    non_spam_data = non_spam_data.sample(n=min(N_SAMPLES, len(non_spam_data)), random_state=RANDOM_STATE)
    print(f"Loaded {len(non_spam_data)} non-spam samples.")
except FileNotFoundError:
     raise FileNotFoundError(f"Non-spam file not found: {NON_SPAM_FILE}")
except Exception as e:
     raise RuntimeError(f"Error loading non-spam data: {e}")


# --- Combine Data ---
print("Combining spam and non-spam data...")
combined_data = pd.concat([spam_data, non_spam_data], ignore_index=True)
# Handle potential NaN values in text column resulting from loading issues or empty comments
combined_data['text'] = combined_data['text'].fillna('')
print(f"Total samples: {len(combined_data)}")
print(f"Class distribution:\n{combined_data['label'].value_counts()}")

# --- Data Splitting ---
print("Splitting data into training and testing sets...")
X = combined_data['text']
y = combined_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# --- Model Training and Evaluation ---
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Multinomial Naive Bayes": MultinomialNB()
}

results = {}

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

print("\n--- Training and Evaluating Models ---")
for name, classifier in classifiers.items():
    print(f"\nTraining {name}...")
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', classifier)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam'])

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    results[name] = {'pipeline': pipeline, 'accuracy': accuracy, 'report': report}

# --- Plot Learning Curves ---
print(f"\nGenerating learning curves and saving to {OUTPUT_PLOT_FILE}...")

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve."""
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score (Accuracy)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt


n_classifiers = len(classifiers)
fig, axes = plt.subplots(1, n_classifiers, figsize=(8 * n_classifiers, 6), sharey=True)
fig.suptitle('Learning Curves for Spam Classifiers', fontsize=16)

if n_classifiers == 1: # Handle case with only one classifier
    axes = [axes]

for i, (name, result) in enumerate(results.items()):
    plot_learning_curve(result['pipeline'], name, X_train, y_train, axes=axes[i], ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig(OUTPUT_PLOT_FILE)
print(f"Learning curves saved to {OUTPUT_PLOT_FILE}")

print("\nScript finished.") 