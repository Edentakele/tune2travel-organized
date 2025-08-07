import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, concatenate_datasets, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os

# --- Config ---
CSV_FILES = [
    'despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv',
    'seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv',
    'generated_fear_v2.csv',
    'generated_anger_disgust.csv'
]
MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
VAL_SIZE = 0.1
SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
OUTPUT_DIR = './emotion_model_output_best_hyper'
LABEL_COL = 'manual_emotion_label_eden'
COMMENT_COL1 = 'comment_no_emojis'
COMMENT_COL2 = 'emoji_text_representation'
TEXT_COL = 'input_text_for_model'

# --- Load and combine data ---
all_dfs = []
for file in CSV_FILES:
    df = pd.read_csv(file)
    # Fill NaN with empty string for emoji_text_representation
    df[COMMENT_COL1] = df[COMMENT_COL1].fillna('')
    df[COMMENT_COL2] = df[COMMENT_COL2].fillna('')
    # Concatenate comment_no_emojis and emoji_text_representation
    df[TEXT_COL] = df[COMMENT_COL1] + ' ' + df[COMMENT_COL2]
    # Drop rows with missing text or label
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[(df[TEXT_COL].str.strip() != '') & (df[LABEL_COL] != '')]
    all_dfs.append(df[[TEXT_COL, LABEL_COL]])
full_df = pd.concat(all_dfs, ignore_index=True)

# --- Encode labels ---
labels = sorted(full_df[LABEL_COL].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
full_df['label'] = full_df[LABEL_COL].map(label2id)

# --- Train/val split ---
train_df, val_df = train_test_split(full_df, test_size=VAL_SIZE, random_state=SEED, stratify=full_df['label'])
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, padding='max_length', max_length=128)
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# --- Training args ---
from sklearn.metrics import accuracy_score, classification_report

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    greater_is_better=True,
    seed=SEED,
    report_to='none',
    fp16=torch.cuda.is_available(),
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- Evaluation ---
metrics = trainer.evaluate()
print('Validation metrics:', metrics)

# --- Classification report ---
preds = trainer.predict(val_ds)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels))

# --- Save final model ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'Model and tokenizer saved to {OUTPUT_DIR}') 