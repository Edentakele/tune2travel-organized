import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
CSV_FILES = [
    'despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv',
    'seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv',
    'generated_fear_v2.csv',
    'generated_anger_disgust.csv'
]
MODEL_DIR = './emotion_model_output_best_hyper'
LABEL_COL = 'manual_emotion_label_eden'
COMMENT_COL1 = 'comment_no_emojis'
COMMENT_COL2 = 'emoji_text_representation'
TEXT_COL = 'input_text_for_model'
BATCH_SIZE = 32

# --- Load data ---
all_dfs = []
for file in CSV_FILES:
    df = pd.read_csv(file)
    df[COMMENT_COL1] = df[COMMENT_COL1].fillna('')
    df[COMMENT_COL2] = df[COMMENT_COL2].fillna('')
    df[TEXT_COL] = df[COMMENT_COL1] + ' ' + df[COMMENT_COL2]
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[(df[TEXT_COL].str.strip() != '') & (df[LABEL_COL] != '')]
    all_dfs.append(df[[TEXT_COL, LABEL_COL]])
full_df = pd.concat(all_dfs, ignore_index=True)

# --- Encode labels ---
labels = sorted(full_df[LABEL_COL].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
full_df['label'] = full_df[LABEL_COL].map(label2id)

# --- Load model and tokenizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

# --- Inference ---
preds = []
true = []
model.eval()
with torch.no_grad():
    for i in range(0, len(full_df), BATCH_SIZE):
        batch_texts = full_df[TEXT_COL].iloc[i:i+BATCH_SIZE].tolist()
        batch_labels = full_df['label'].iloc[i:i+BATCH_SIZE].tolist()
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        for k in enc:
            enc[k] = enc[k].to(device)
        outputs = model(**enc)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)
        true.extend(batch_labels)

# --- Metrics ---
print(classification_report(true, preds, labels=list(range(len(labels))), target_names=labels))
print(f"Accuracy: {accuracy_score(true, preds):.4f}")

# --- Confusion Matrix Visualization ---
cm = confusion_matrix(true, preds, labels=list(range(len(labels))))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show() 