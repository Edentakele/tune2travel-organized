import pandas as pd
import numpy as np
import glob
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TextClassificationPipeline
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel, Features, Value
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Configuration ---
SPAM_DATA_DIR = '/run/media/ottobeeth/362A4A2E2A49EC05/Users/ottobeeth/courses/tune2travel/emotion_analysis/spam_detector_2000/dataset_approach'
NON_SPAM_FILE = '/run/media/ottobeeth/362A4A2E2A49EC05/Users/ottobeeth/courses/tune2travel/data/yearly_data/despa_2018_en_nospam_labelled_default.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "bert-base-uncased" # Or a more specific spam detection model like "unitary/toxic-bert"
OUTPUT_DIR = './hf_spam_classifier_results'
LOGGING_DIR = './hf_spam_classifier_logs'
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 16

# --- Load Data using Hugging Face datasets --- 
print("Loading data using Hugging Face datasets...")

# Load Spam Data
spam_files_list = glob.glob(os.path.join(SPAM_DATA_DIR, '*.csv'))
if not spam_files_list:
    raise FileNotFoundError(f"No CSV files found in {SPAM_DATA_DIR}.")

spam_datasets = []
for f in spam_files_list:
    try:
        # Load dataset, rename columns, select necessary ones, filter for spam
        ds = load_dataset('csv', data_files=f, split='train')
        # Check for expected columns before renaming
        if 'CONTENT' in ds.column_names and 'CLASS' in ds.column_names:
            ds = ds.rename_columns({'CONTENT': 'text', 'CLASS': 'label'}) # Rename
            ds = ds.select_columns(['text', 'label'])
            ds = ds.filter(lambda example: example['label'] == 1) # Filter for spam (label=1)
            if len(ds) > 0:
                spam_datasets.append(ds)
            else:
                print(f"Warning: No spam samples found in {f} after filtering.")
        else:
            print(f"Warning: Skipping file {f} due to missing 'CONTENT' or 'CLASS' columns.")
    except Exception as e:
        print(f"Error loading or processing file {f} with datasets: {e}")

if not spam_datasets:
    raise ValueError("No valid spam data could be loaded using datasets library.")

spam_data_full = concatenate_datasets(spam_datasets)
print(f"Loaded {len(spam_data_full)} spam samples using datasets.")

# Load Non-Spam Data
try:
    non_spam_dataset = load_dataset('csv', data_files=NON_SPAM_FILE, split='train')
    # Check for the 'comment' column instead of 'processed_comment'
    if 'comment' not in non_spam_dataset.column_names:
         raise ValueError(f"Missing 'comment' column in {NON_SPAM_FILE}")
    # Rename 'comment' to 'text'
    non_spam_dataset = non_spam_dataset.rename_column('comment', 'text')
    non_spam_dataset = non_spam_dataset.select_columns(['text'])
    # Add the label column (0 for non-spam)
    non_spam_dataset = non_spam_dataset.map(lambda example: {'label': 0})
    
    # --- Sample Non-Spam Data ---
    num_non_spam_samples = 2000 # Define the number of non-spam samples to keep
    if len(non_spam_dataset) > num_non_spam_samples:
        print(f"Sampling {num_non_spam_samples} non-spam samples from {len(non_spam_dataset)}...")
        non_spam_dataset = non_spam_dataset.shuffle(seed=RANDOM_STATE).select(range(num_non_spam_samples))
    else:
        print(f"Warning: Available non-spam samples ({len(non_spam_dataset)}) is less than or equal to the desired sample size ({num_non_spam_samples}). Using all available non-spam samples.")

    print(f"Using {len(non_spam_dataset)} non-spam samples.")

except FileNotFoundError:
     raise FileNotFoundError(f"Non-spam file not found: {NON_SPAM_FILE}")
except Exception as e:
     raise RuntimeError(f"Error loading or sampling non-spam data using datasets: {e}")


# --- Combine Data --- 
print("Combining spam and non-spam datasets...")

combined_dataset = concatenate_datasets([spam_data_full, non_spam_dataset])

# Ensure text is string and handle potential nulls
combined_dataset = combined_dataset.map(lambda example: {'text': str(example['text']) if example['text'] is not None else ''})

# Shuffle the combined dataset
combined_dataset = combined_dataset.shuffle(seed=RANDOM_STATE)

# --- Cast label column to ClassLabel --- 
print("Casting label column to ClassLabel type for stratification...")
# Define the features, specifically casting 'label' to ClassLabel
# Ensure the label values are suitable for casting (e.g., int)
# First, ensure the label column is integer type if it's not already
# combined_dataset = combined_dataset.cast_column("label", Value("int64")) # Uncomment if needed, map usually infers int

new_features = Features({
    'text': Value('string'),  # Keep text as string
    'label': ClassLabel(names=['Non-Spam', 'Spam']) # Define label as ClassLabel
})
combined_dataset = combined_dataset.cast(new_features)
print("Label column casted.")

print(f"Total samples: {len(combined_dataset)}")

# --- Split Data using datasets --- 
print("Splitting dataset into training and testing sets...")
# Now stratification should work
dataset_split = combined_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_STATE, stratify_by_column='label')

# Rename the default 'train' and 'test' splits if desired, or use them directly
train_dataset = dataset_split['train']
test_dataset = dataset_split['test']

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# --- Tokenization --- 
print(f"Loading tokenizer for model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizing datasets using .map()...")
def tokenize_function(examples):
    # Tokenize the text. The tokenizer handles padding and truncation.
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Apply tokenization to the datasets batched=True speeds it up
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove the original 'text' column as it's no longer needed after tokenization
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

# Set the format for PyTorch
tokenized_train_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")

# --- Create PyTorch Datasets ---  <- No longer needed! The Trainer accepts tokenized_datasets
# class SpamDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
# 
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
# 
#     def __len__(self):
#         return len(self.labels)
# 
# train_dataset = SpamDataset(train_encodings, train_labels)
# test_dataset = SpamDataset(test_encodings, test_labels)

# --- Load Pre-trained Model ---
print(f"Loading model: {MODEL_NAME}")
# Assuming binary classification (spam/non-spam), num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --- Define Evaluation Metrics ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Non-Spam', 'Spam'], output_dict=True, zero_division=0)
    return {
        'accuracy': accuracy,
        'precision_spam': report['Spam']['precision'],
        'recall_spam': report['Spam']['recall'],
        'f1_spam': report['Spam']['f1-score'],
        'precision_nonspam': report['Non-Spam']['precision'],
        'recall_nonspam': report['Non-Spam']['recall'],
        'f1_nonspam': report['Non-Spam']['f1-score'],
    }

# --- Setup Trainer ---
print("Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,
    logging_steps=100,
    report_to="none", # Disable wandb/tensorboard integration for simplicity
    fp16=torch.cuda.is_available(), # Enable mixed precision training if GPU is available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# --- Train Model ---
print("Starting training...")
train_result = trainer.train()
print("Training finished.")

# Save training metrics
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

# --- Evaluate Model ---
print("Evaluating model on the test set...")
eval_metrics = trainer.evaluate()
print("Evaluation finished.")

# Save evaluation metrics
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

print("--- Evaluation Results ---")
for key, value in eval_metrics.items():
    print(f"{key}: {value:.4f}")

# --- Save Model and Tokenizer ---
print(f"Saving fine-tuned model and tokenizer to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- Classify TEST SET and Save --- 
print("\n--- Classifying Test Set and Saving Predictions --- ")

# Load the fine-tuned model pipeline
print("Loading fine-tuned pipeline...")
# Ensure the model loaded is the one from the trainer (which should be the best one)
pipe = TextClassificationPipeline(model=trainer.model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# The test_dataset variable still holds the split data BEFORE tokenization
# It should have 'text' and 'label' (original label) columns
print(f"Starting prediction on the test set ({len(test_dataset)} samples)...")

if 'text' not in test_dataset.column_names:
    print("Error: 'text' column not found in test_dataset. Cannot perform prediction.")
else:
    # Predict on the text from the test_dataset
    # Add truncation=True to handle potential edge cases even in test set
    print("Generating predictions for the test set...")
    # Generate all predictions first and store in a list
    # Add return_all_scores=True to get scores for both labels
    all_test_predictions = list(pipe(test_dataset["text"], batch_size=PER_DEVICE_EVAL_BATCH_SIZE, truncation=True, return_all_scores=True))
    print("Predictions generated.")

    test_predictions_results = []
    count = 0
    # Loop through indices of the test dataset
    print("Processing predictions...")
    for i in range(len(test_dataset)):
        # Get the prediction output for the current index
        prediction_output = all_test_predictions[i]

        # Get original text and label from the test_dataset
        original_comment = test_dataset[i]['text']
        # Original label is stored as an integer (0 or 1) after ClassLabel casting
        original_label_id = test_dataset[i]['label']
        original_label_name = "Spam" if original_label_id == 1 else "Non-Spam"
        
        # Process prediction output (should be a list of dicts like [{'label': '...', 'score': ...}, ...])
        try:
            best_label = max(prediction_output, key=lambda x: x['score'])
            pred_label_id = int(best_label['label'].split('_')[-1])
            pred_label_name = "Spam" if pred_label_id == 1 else "Non-Spam"
            pred_score = best_label['score']
        except (TypeError, KeyError, IndexError) as e:
            print(f"Warning: Could not parse prediction output for item {i}: {prediction_output}. Error: {e}. Skipping.")
            # Assign default/error values if parsing fails
            pred_label_name = "Error"
            pred_score = 0.0
        
        test_predictions_results.append({
            "comment": original_comment,
            "original_label": original_label_name, 
            "predicted_label": pred_label_name,
            "confidence": pred_score
        })
        count += 1
        if count % 100 == 0:
             print(f"  Processed {count}/{len(test_dataset)} test samples...")

    print(f"Finished processing {len(test_predictions_results)} test comments.")

    # --- Filter and Save Test Results --- 
    print("Filtering and saving test set predictions...")
    
    # Convert results to DataFrame
    test_results_df = pd.DataFrame(test_predictions_results)
    
    # Filter based on predicted label
    test_predicted_spam_df = test_results_df[test_results_df["predicted_label"] == "Spam"]
    test_predicted_non_spam_df = test_results_df[test_results_df["predicted_label"] == "Non-Spam"]
    
    # Save to CSV
    output_test_spam_file = "test_predictions_spam.csv"
    output_test_non_spam_file = "test_predictions_non_spam.csv"
    
    test_predicted_spam_df.to_csv(output_test_spam_file, index=False)
    print(f"Saved {len(test_predicted_spam_df)} test comments predicted as SPAM to {output_test_spam_file}")
    
    test_predicted_non_spam_df.to_csv(output_test_non_spam_file, index=False)
    print(f"Saved {len(test_predicted_non_spam_df)} test comments predicted as NON-SPAM to {output_test_non_spam_file}")

print("\nScript finished.") 