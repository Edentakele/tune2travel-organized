import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import itertools

# --- Config ---
CSV_FILES = [
    'despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv',
    'seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv',
    'generated_fear_v2.csv',
    'generated_anger_disgust.csv'
]
LABEL_COL = 'manual_emotion_label_eden'
COMMENT_COL1 = 'comment_no_emojis'
COMMENT_COL2 = 'emoji_text_representation'
TEXT_COL = 'input_text_for_model'
VAL_SIZE = 0.15
SEED = 42
OUTPUT_DIR = './emotion_model_hyper_search'

# --- Hyperparameter grid ---
MODELS = [
    'j-hartmann/emotion-english-distilroberta-base',
]
LEARNING_RATES = [2e-5, 5e-5]
BATCH_SIZES = [16, 32]
EPOCHS = [3, 4]
WEIGHT_DECAYS = [0.01, 0.001]

# --- Load and combine data ---
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

# --- Train/val split ---
train_df, val_df = train_test_split(full_df, test_size=VAL_SIZE, random_state=SEED, stratify=full_df['label'])
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# --- Tokenization function ---
def tokenize(batch, tokenizer):
    return tokenizer(batch[TEXT_COL], truncation=True, padding='max_length', max_length=128)

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# --- Hyperparameter search ---
results = []
best_f1 = -1
best_params = None
best_model_dir = None
run_id = 0
for model_name, lr, bs, n_epochs, wd in itertools.product(MODELS, LEARNING_RATES, BATCH_SIZES, EPOCHS, WEIGHT_DECAYS):
    run_id += 1
    print(f"\nRun {run_id}: Model={model_name}, LR={lr}, BS={bs}, Epochs={n_epochs}, WD={wd}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_tok = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
        val_tok = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/run_{run_id}_{model_name.replace('/', '_')}",
            eval_strategy='epoch',
            save_strategy='no',
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            num_train_epochs=n_epochs,
            weight_decay=wd,
            load_best_model_at_end=False,
            seed=SEED,
            report_to='none',
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        preds = trainer.predict(val_tok)
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        print(f"  Validation accuracy: {acc:.4f}, weighted f1: {f1:.4f}")
        results.append({
            'run_id': run_id,
            'model': model_name,
            'learning_rate': lr,
            'batch_size': bs,
            'epochs': n_epochs,
            'weight_decay': wd,
            'val_accuracy': acc,
            'val_f1': f1
        })
        if f1 > best_f1:
            best_f1 = f1
            best_params = (model_name, lr, bs, n_epochs, wd)
            best_model_dir = f"{OUTPUT_DIR}/best_model"
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
    except Exception as e:
        print(f"  Error in run {run_id}: {e}")

# --- Save results ---
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/hyperparameter_search_results.csv", index=False)
print("\nBest run:")
print(f"  Model: {best_params[0]}\n  Learning rate: {best_params[1]}\n  Batch size: {best_params[2]}\n  Epochs: {best_params[3]}\n  Weight decay: {best_params[4]}\n  Best weighted f1: {best_f1:.4f}")
print(f"Best model saved to: {best_model_dir}") 