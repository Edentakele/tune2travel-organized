import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, accuracy_score
import os

# --- Config ---
CSV_FILES = [
    'despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv',
    'seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv',
]
MODEL_DIR = './emotion_model_output'
LABEL_COL = 'manual_emotion_label_eden'
COMMENT_COL1 = 'comment_no_emojis'
COMMENT_COL2 = 'emoji_text_representation'
TEXT_COL = 'input_text_for_model'
BATCH_SIZE = 32

# --- Load data ---
all_dfs = []
for file in CSV_FILES:
    df = pd.read_csv(file)
    # Fill NaN with empty string for emoji_text_representation
    df[COMMENT_COL1] = df[COMMENT_COL1].fillna('')
    df[COMMENT_COL2] = df[COMMENT_COL2].fillna('')
    # Concatenate comment_no_emojis and emoji_text_representation
    df[TEXT_COL] = df[COMMENT_COL1] + ' ' + df[COMMENT_COL2]
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[(df[TEXT_COL].str.strip() != '') & (df[LABEL_COL] != '')]
    all_dfs.append(df[[TEXT_COL, LABEL_COL]])
full_df = pd.concat(all_dfs, ignore_index=True)

# --- Label encoding ---
labels = sorted(full_df[LABEL_COL].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
full_df['label'] = full_df[LABEL_COL].map(label2id)

# --- Load model and tokenizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# --- Inference ---
preds = []
with torch.no_grad():
    for i in range(0, len(full_df), BATCH_SIZE):
        batch_texts = full_df[TEXT_COL].iloc[i:i+BATCH_SIZE].tolist()
        enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)

# --- Metrics ---
y_true = full_df['label'].values
y_pred = np.array(preds)
print('Accuracy:', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels)) 