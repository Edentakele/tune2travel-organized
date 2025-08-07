import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
import torch
from sklearn.metrics import classification_report, accuracy_score

# --- Config ---
CSV_FILES = [
    'despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv',
    'seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv',
]
LABEL_COL = 'manual_emotion_label_eden'
COMMENT_COL1 = 'comment_no_emojis'
COMMENT_COL2 = 'emoji_text_representation'
TEXT_COL = 'input_text_for_model'
BATCH_SIZE = 32

# Models to compare (model_name, is_t5)
# Only include models with the exact same 7-label mapping as the dataset.
MODELS = [
    # This model's id2label matches: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    ('j-hartmann/emotion-english-distilroberta-base', False),
    # Other public models (e.g., cardiffnlp/twitter-roberta-base-emotion, bhadresh-savani/electra-base-emotion, mrm8488/t5-base-finetuned-emotion)
    # are excluded because their label sets do not match the dataset and would result in poor or misleading evaluation.
]

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

# --- Label encoding ---
labels = sorted(full_df[LABEL_COL].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
full_df['label'] = full_df[LABEL_COL].map(label2id)

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Inference function ---
def predict_with_model(model_name, is_t5):
    print(f'\nEvaluating model: {model_name}')
    if is_t5:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(full_df), BATCH_SIZE):
                batch_texts = full_df[TEXT_COL].iloc[i:i+BATCH_SIZE].tolist()
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                outputs = model.generate(**inputs, max_length=16)
                batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # Map T5 outputs to label ids (case-insensitive match)
                batch_pred_ids = [label2id.get(pred.strip().lower(), -1) for pred in batch_preds]
                preds.extend(batch_pred_ids)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        # Try to get model's id2label if available
        model_id2label = getattr(model.config, 'id2label', None)
        preds = []
        with torch.no_grad():
            for i in range(0, len(full_df), BATCH_SIZE):
                batch_texts = full_df[TEXT_COL].iloc[i:i+BATCH_SIZE].tolist()
                enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                batch_pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
                # If model's id2label doesn't match our label2id, try to map
                if model_id2label:
                    batch_pred_labels = [model_id2label.get(int(idx), '').lower() for idx in batch_pred_ids]
                    batch_pred_ids = [label2id.get(lbl, -1) for lbl in batch_pred_labels]
                preds.extend(batch_pred_ids)
    # Remove samples where prediction failed (-1)
    y_true = full_df['label'].values
    y_pred = np.array(preds)
    valid = y_pred != -1
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels))
    return accuracy_score(y_true, y_pred)

# --- Run experiment ---
results = {}
for model_name, is_t5 in MODELS:
    try:
        acc = predict_with_model(model_name, is_t5)
        results[model_name] = acc
    except Exception as e:
        print(f'Error evaluating {model_name}: {e}')

print('\n--- Summary ---')
for model_name, acc in results.items():
    print(f'{model_name}: Accuracy = {acc:.4f}') 