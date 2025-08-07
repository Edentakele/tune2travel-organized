import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import os # Import os for path joining
import csv # Import csv for writing results progressively
import json # Import json for progress tracking

# 1. Define the 7 target emotions (including neutral as per the model)
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'] # Added neutral
NUM_LABELS = len(EMOTIONS)
# Note: The actual model might have a different internal mapping.
# We are defining it here based on common practice, but the model's config
# will be the source of truth after loading. We adjust it during loading.
ID2LABEL = {i: label for i, label in enumerate(EMOTIONS)}
LABEL2ID = {label: i for i, label in enumerate(EMOTIONS)}

# 2. Set Model Name
# Using the model identified from Hugging Face Hub
MODEL_NAME = "j-hartmann/emotion-english-roberta-large"
OUTPUT_CSV_FILENAME = "results/emotion_analysis_output.csv" # Define output file path
PROGRESS_FILE = "results/emotion_analysis_progress.json" # Define progress file path

# 3. Load Model and Tokenizer
try:
    print(f"Loading tokenizer for model: {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading model: {MODEL_NAME}...")
    # The model config (id2label, label2id) will be loaded from the checkpoint
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
    # Verify and update our local mappings if needed (optional but good practice)
    # Check if the number of labels matches
    if model.config.num_labels != NUM_LABELS:
         print(f"Warning: Model config has {model.config.num_labels} labels, but {NUM_LABELS} were expected based on EMOTIONS list.")
         # Adjust based on model's config if necessary, or raise an error
    print("Model and tokenizer loaded successfully.")
    # Update local maps based on loaded model config
    ID2LABEL = model.config.id2label
    LABEL2ID = model.config.label2id
    EMOTIONS = list(ID2LABEL.values()) # Ensure our EMOTIONS list matches model
    NUM_LABELS = model.config.num_labels
    print(f"Model configured for {NUM_LABELS} labels: {EMOTIONS}")

except OSError as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    print("Please ensure the model name is correct and you have internet connectivity.")
    print("You might need to install git-lfs and be logged into Hugging Face Hub CLI.")
    model = None
    tokenizer = None
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    model = None
    tokenizer = None


# 4. Load and Preprocess Data from results/lr_output.csv
def load_and_preprocess_data(data_path="results/lr_output.csv"):
    """
    Loads comments from the specified CSV file, filters for 'is_spam_prediction_lr' == 0,
    and returns the list of comments.
    """
    # Construct the full path relative to the script's location or workspace root
    # Assuming the script is run from the workspace root or results/ is accessible
    full_path = data_path # Use relative path directly if running from workspace root

    print(f"Attempting to load data from: {full_path}")

    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}")
        print("Please ensure 'results/lr_output.csv' exists in the correct location.")
        return [] # Return empty list if file not found

    try:
        df = pd.read_csv(full_path)
        print(f"Successfully loaded {full_path}. Columns: {df.columns.tolist()}")

        # Define the required columns
        label_col = 'is_spam_prediction_lr' # Use the correct column name
        comment_col = 'original_text'

        # Check if the required columns exist
        if label_col not in df.columns or comment_col not in df.columns:
            print(f"Error: Required columns ('{label_col}', '{comment_col}') not found in {full_path}")
            # If 'labelled' exists maybe fallback or suggest?
            if 'labelled' in df.columns:
                 print(f"Found 'labelled' column instead of '{label_col}'. Adjust script if 'labelled' is the correct column.")
            return []

        # Filter rows where the label column is 0
        filtered_df = df[df[label_col] == 0]
        print(f"Found {len(filtered_df)} comments with {label_col} == 0.")

        # Extract the 'comment' column and convert to list
        # Handle potential NaN values by converting to string and replacing 'nan'
        texts = filtered_df[comment_col].astype(str).replace('nan', '').tolist()

        # Remove any empty strings that might result from NaN conversion
        texts = [text for text in texts if text]

        print(f"Prepared {len(texts)} non-empty comments for analysis.")
        return texts

    except pd.errors.EmptyDataError:
        print(f"Error: The file {full_path} is empty.")
        return []
    except Exception as e:
        print(f"Error reading or processing CSV file {full_path}: {e}")
        return []

# 5. Perform Emotion Analysis
def analyze_emotions(texts, output_filename, progress_filename, start_count, total_original_count):
    """
    Analyzes the emotion of a list of texts (those not yet processed),
    writes results incrementally, and updates a progress file after each batch.
    Returns the total number of texts processed *in this run*.
    """
    if not model or not tokenizer:
        print("Model or tokenizer not loaded. Cannot perform analysis.")
        return 0 # Return 0 processed items

    # Ensure the results directory exists (for output csv and progress json)
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return 0

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Note: len(texts) is the number to process *in this run*
    print(f"Performing inference on {len(texts)} remaining texts using {device}...")
    print(f"Results will be saved progressively to: {output_filename}")
    print(f"Progress will be tracked in: {progress_filename}")

    batch_size = 16 # Process in batches
    processed_in_this_run = 0

    try:
        # Open the output file in append mode
        with open(output_filename, 'a', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)

            # Write header only if we are starting from the beginning (start_count == 0)
            # and the output file is actually empty.
            is_new_file = f_out.tell() == 0
            if start_count == 0 and is_new_file:
                writer.writerow(["text", "predicted_emotion"]) # Add other columns if needed
                print("Writing header to new output file.")
            elif start_count > 0 and is_new_file:
                # Warn if progress exists but file is missing/empty - might indicate an issue
                print(f"Warning: Progress file indicates {start_count} items processed, but output file {output_filename} is empty or missing. Appending results anyway.")
                # Consider writing header here too if you want it in this case
                # writer.writerow(["text", "predicted_emotion"])

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                predicted_class_ids = torch.argmax(logits, dim=-1).cpu().tolist()

                batch_results = []
                for j, text in enumerate(batch_texts):
                    pred_id = predicted_class_ids[j]
                    predicted_label = model.config.id2label[pred_id]
                    row_data = [text, predicted_label]
                    batch_results.append(row_data)

                # 1. Write the batch results to the CSV
                writer.writerows(batch_results)
                f_out.flush() # Ensure data is written to disk

                processed_in_this_run += len(batch_texts)
                current_total_processed = start_count + processed_in_this_run

                # 2. Update the progress file
                try:
                    with open(progress_filename, 'w', encoding='utf-8') as f_prog:
                        json.dump({"processed_count": current_total_processed}, f_prog)
                except Exception as e_prog:
                    print(f"\nWarning: Could not write progress to {progress_filename}: {e_prog}")
                    # Continue processing, but progress won't be saved for this batch

                # 3. Print progress update (relative to total original count)
                print(f"  Processed and saved {current_total_processed} / {total_original_count} texts...")

    except Exception as e:
        print(f"\nAn error occurred during inference or file writing: {e}")
        # Return the count processed *in this run* so far
        return processed_in_this_run

    print(f"\nInference completed for this run. {processed_in_this_run} new results saved to {output_filename}.")
    return processed_in_this_run

# 6. Main Execution Block
if __name__ == "__main__":
    print("Starting Emotion Analysis Script...")

    # Ensure model is loaded before proceeding
    if model and tokenizer:
        # Load all potential input data
        all_input_texts = load_and_preprocess_data() # Path is hardcoded in function
        total_texts = len(all_input_texts)

        if total_texts == 0:
            print("\nNo valid input texts found or loaded. Ensure 'results/lr_output.csv' exists, is valid, and contains non-spam comments.")
        else:
            # Load progress
            processed_count = 0
            if os.path.exists(PROGRESS_FILE):
                try:
                    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f_prog:
                        progress_data = json.load(f_prog)
                        if isinstance(progress_data, dict) and "processed_count" in progress_data and isinstance(progress_data["processed_count"], int):
                            processed_count = progress_data["processed_count"]
                            # Sanity check: ensure processed_count isn't higher than total texts
                            if processed_count > total_texts:
                                print(f"Warning: Progress file indicates {processed_count} processed, but only {total_texts} found in input. Resetting progress.")
                                processed_count = 0
                            elif processed_count < 0:
                                print(f"Warning: Progress file contains negative count ({processed_count}). Resetting progress.")
                                processed_count = 0
                            else:
                                print(f"Loaded progress: {processed_count} texts already processed.")
                        else:
                             print(f"Warning: Invalid format in progress file {PROGRESS_FILE}. Starting from beginning.")
                             processed_count = 0
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {PROGRESS_FILE}. Starting from beginning.")
                    processed_count = 0
                except Exception as e:
                    print(f"Warning: Error reading progress file {PROGRESS_FILE}: {e}. Starting from beginning.")
                    processed_count = 0
            else:
                print("No progress file found. Starting from beginning.")

            # Check if already completed
            if processed_count >= total_texts:
                print(f"\nAnalysis previously completed. All {total_texts} texts are processed according to {PROGRESS_FILE}.")
                print(f"Output should be in {OUTPUT_CSV_FILENAME}.")
            else:
                # Determine texts for this run
                texts_to_process = all_input_texts[processed_count:]
                remaining_count = len(texts_to_process)
                print(f"Preparing to process {remaining_count} remaining texts (out of {total_texts} total).")

                # Define output file path
                output_file = OUTPUT_CSV_FILENAME

                # Optional: Delete existing output file if starting fresh (processed_count == 0)
                # if processed_count == 0 and os.path.exists(output_file):
                #     print(f"Starting fresh, deleting existing output file: {output_file}")
                #     try:
                #         os.remove(output_file)
                #     except OSError as e:
                #         print(f"Error deleting existing file {output_file}: {e}. Appending instead.")

                # Analyze emotions and save progressively
                num_newly_processed = analyze_emotions(texts_to_process, output_file, PROGRESS_FILE, processed_count, total_texts)

                final_processed_count = processed_count + num_newly_processed

                # Print summary based on processed count
                print(f"\nEmotion analysis run finished.")
                print(f"Processed {num_newly_processed} texts in this run.")
                print(f"Total processed count is now {final_processed_count} out of {total_texts}.")

                if final_processed_count == total_texts:
                    print(f"All texts have been processed successfully!")
                    # Optional: Clean up progress file
                    # try:
                    #     os.remove(PROGRESS_FILE)
                    #     print(f"Removed progress file: {PROGRESS_FILE}")
                    # except OSError as e:
                    #     print(f"Could not remove progress file {PROGRESS_FILE}: {e}")
                else:
                    print(f"Run interrupted or incomplete. Re-run the script to continue processing the remaining {total_texts - final_processed_count} texts.")

                # Show sample from output file
                if os.path.exists(output_file):
                    try:
                        print("\nSample from output file:")
                        # Use tail or read last lines if file is large? For now, read head.
                        sample_df = pd.read_csv(output_file, nrows=5)
                        print(sample_df.to_string(index=False))
                    except Exception as e:
                        print(f"Could not read sample from output file {output_file}: {e}")
                else:
                    print(f"Output file {output_file} not found or not yet created.")

    else:
        print("\nSetup incomplete: Model or tokenizer failed to load. Cannot proceed.")

    print("Emotion Analysis Script Finished.")