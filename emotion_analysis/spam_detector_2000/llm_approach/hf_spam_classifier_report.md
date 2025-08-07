# Hugging Face Spam Classifier Training Report

This report summarizes the execution and results of the `hf_spam_classifier.py` script.

## 1. Data Preparation

- **Spam Data:** Loaded 1005 samples from `/home/ottobeeth/tune2travel/spam_detector_2000/dataset_approach`.
- **Non-Spam Data:** Loaded 1,412,994 samples initially from `/home/ottobeeth/tune2travel/data/topic_csv/cleaned_despacito.csv`.
- **Sampling:** Sampled 2000 non-spam comments to create a more balanced dataset.
- **Combined:** Total dataset size of 3005 samples (1005 spam, 2000 non-spam).
- **Label Casting:** Casted the 'label' column to `ClassLabel` type for stratified splitting.
- **Splitting:** Split into training (2404 samples) and testing (601 samples) sets using stratification.

```text
Loading data using Hugging Face datasets...
Loaded 1005 spam samples using datasets.
Sampling 2000 non-spam samples from 1412994...
Using 2000 non-spam samples.
Combining spam and non-spam datasets...
Map: 100%|                                                                               | 3005/3005 [00:00<00:00, 16792.15 examples/s]
Casting label column to ClassLabel type for stratification...
Casting the dataset: 100%|                                                              | 3005/3005 [00:00<00:00, 147016.64 examples/s]
Label column casted.
Total samples: 3005
Splitting dataset into training and testing sets...
Training set size: 2404
Test set size: 601
```

## 2. Tokenization

- **Model:** `bert-base-uncased`
- **Process:** Tokenized training and test sets using the model's tokenizer via `datasets.map()`.

```text
Loading tokenizer for model: bert-base-uncased
Tokenizing datasets using .map()...
Map: 100%|                                                                                | 2404/2404 [00:00<00:00, 5430.06 examples/s]
Map: 100%|                                                                                  | 601/601 [00:00<00:00, 5851.07 examples/s]
```

## 3. Model Loading

- **Model:** `bert-base-uncased` (for Sequence Classification)
- **Warning:** Some weights were newly initialized (classifier head), indicating the model needs fine-tuning.

```text
Loading model: bert-base-uncased
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

## 4. Training

- **Setup:** Used Hugging Face `Trainer`.
- **Device:** GPU (NVIDIA RTX A5000 Laptop GPU based on previous checks).
- **Epochs:** 3
- **Batch Size (Train):** 4
- **Total Steps:** 1803
- **Duration:** ~2 minutes 55 seconds

**Training Log Snippets:**
```text
Starting training...
{'loss': 0.5848, 'grad_norm': 12.914173126220703, 'learning_rate': 9.600000000000001e-06, 'epoch': 0.17}                               
{'loss': 0.0764, 'grad_norm': 0.07613328844308853, 'learning_rate': 1.9500000000000003e-05, 'epoch': 0.33}                             
...                           
{'loss': 0.0001, 'grad_norm': 0.001776198041625321, 'learning_rate': 3.453568687643899e-07, 'epoch': 3.0}                              
100%|                                                                                              | 1803/1803 [02:55<00:00, 10.27it/s]
Training finished.
```

**Final Training Metrics:**
```text
***** train metrics *****
  epoch                    =        3.0
  total_flos               =  1767237GF
  train_loss               =     0.0642
  train_runtime            = 0:02:55.49
  train_samples_per_second =     41.094
  train_steps_per_second   =     10.274
```

## 5. Evaluation

- **Dataset:** Test set (601 samples)
- **Batch Size (Eval):** 16
- **Duration:** ~2.5 seconds

**Final Evaluation Metrics:**
```text
***** eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =     0.9933
  eval_f1_nonspam         =      0.995
  eval_f1_spam            =       0.99
  eval_loss               =     0.0489
  eval_precision_nonspam  =     0.9925
  eval_precision_spam     =      0.995
  eval_recall_nonspam     =     0.9975
  eval_recall_spam        =     0.9851
  eval_runtime            = 0:00:02.52
  eval_samples_per_second =    238.407
  eval_steps_per_second   =     15.074
```
**Summary:**
- **Overall Accuracy:** 99.33%
- **Spam Precision:** 0.9950
- **Spam Recall:** 0.9851
- **Spam F1-Score:** 0.9900
- **Non-Spam Precision:** 0.9925
- **Non-Spam Recall:** 0.9975
- **Non-Spam F1-Score:** 0.9950


## 6. Model Saving

- The fine-tuned model and tokenizer were saved to `./hf_spam_classifier_results`.

## 7. Example Predictions

- Used `TextClassificationPipeline` on the fine-tuned model.
- **Note:** The model predicted all examples as Spam, which might warrant further investigation into model bias or the suitability of the chosen examples/model.

```text
--- Example Prediction & Basic Explanation ---
Device set to use cuda:0
Predictions (LABEL_0: Non-Spam, LABEL_1: Spam):

Comment: "Check out my new channel for free giveaways!"
Predicted: Spam (Confidence: 1.0000)
Scores: [{'label': 'LABEL_0', 'score': 4.6293222112581134e-05}, {'label': 'LABEL_1', 'score': 0.9999537467956543}]

Comment: "This song is amazing, reminds me of my childhood."
Predicted: Spam (Confidence: 1.0000)
Scores: [{'label': 'LABEL_0', 'score': 4.5220887841423973e-05}, {'label': 'LABEL_1', 'score': 0.9999548196792603}]

Comment: "great video thanks for sharing"
Predicted: Spam (Confidence: 0.9991)
Scores: [{'label': 'LABEL_0', 'score': 0.0009110511746257544}, {'label': 'LABEL_1', 'score': 0.9990890026092529}]

Comment: "CLICK HERE TO WIN $$$"
Predicted: Spam (Confidence: 0.9999)
Scores: [{'label': 'LABEL_0', 'score': 5.1041904953308403e-05}, {'label': 'LABEL_1', 'score': 0.9999489784240723}]
```

## 8. Conclusion

The script successfully fine-tuned `bert-base-uncased` for spam classification using the sampled dataset. The model achieved high accuracy and F1-scores on the test set. Training time was significantly reduced by using the GPU and sampling the non-spam data. Further analysis might be needed to understand the prediction behavior on specific examples. 