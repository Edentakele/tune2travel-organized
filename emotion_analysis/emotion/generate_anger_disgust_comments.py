import pandas as pd
import random
import uuid
from datetime import datetime, timezone

# --- Configuration ---
NUM_PER_EMOTION = 50
OUTPUT_FILENAME = "generated_anger_disgust_comments_for_training.csv" # Will be in workspace root

MANUALLY_LABELLED_FILES_FOR_IDS = [
    "emotion_analysis/emotion/manually_labelled/despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv",
    "emotion_analysis/emotion/manually_labelled/seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv",
    "emotion_analysis/emotion/generated_fear_v2.csv" # Updated path for fear comments
]
NO_SPAM_FILES_FOR_METADATA = [
    "emotion_analysis/emotion/despa_kJQP7kiw5Fk_comments_no_spam.csv",
    "emotion_analysis/emotion/seeyou_RgKAFK5djSk_comments_no_spam.csv"
]

# --- Keyword and Template Definitions ---

ANGER_KEYWORDS = [
    "furious", "angry", "mad", "rage", "irritated", "pissed off", "infuriated", "livid",
    "outraged", "annoyed", "exasperated", "frustrated", "incensed", "enraged", "steaming"
]
DISGUST_KEYWORDS = [
    "disgusting", "gross", "vile", "repulsive", "sickening", "revolting", "nauseating",
    "yucky", "eww", "horrible", "appalling", "offensive", "obnoxious", "foul", "nasty"
]

# Templates designed for potential contrast, though output is independent
# {SITUATION_NOUN_PHRASE} can be things like "this behavior", "that comment", "the current state of things", "what happened"
# {ACTION_VERB_PHRASE} can be things like "makes me feel", "is simply", "is truly", "I find it"
# {REASON_CLAUSE} can be things like "because it's unfair", "given the impact", "after all that", "without any consideration"
# {OBJECT_OF_EMOTION} can be things like "that attitude", "this mess", "the hypocrisy", "such a statement"

ANGER_TEMPLATES = [
    "I'm so {keyword} about {SITUATION_NOUN_PHRASE}, it's unbelievable!",
    "This {SITUATION_NOUN_PHRASE} {ACTION_VERB_PHRASE} absolutely {keyword}.",
    "Honestly, {SITUATION_NOUN_PHRASE} makes me {keyword} {REASON_CLAUSE}.",
    "It's hard not to be {keyword} when you see {SITUATION_NOUN_PHRASE}.",
    "My blood boils! {SITUATION_NOUN_PHRASE} is just {keyword}.",
    "{OBJECT_OF_EMOTION} is {keyword}, and something needs to be done.",
    "I cannot believe {SITUATION_NOUN_PHRASE}. I am utterly {keyword}.",
    "To say I'm {keyword} about {OBJECT_OF_EMOTION} is an understatement.",
    "This situation with {OBJECT_OF_EMOTION} has made me {keyword}.",
    "I'm feeling incredibly {keyword} towards {OBJECT_OF_EMOTION} right now."
]

DISGUST_TEMPLATES = [
    "Frankly, {SITUATION_NOUN_PHRASE} is utterly {keyword}.",
    "{SITUATION_NOUN_PHRASE} {ACTION_VERB_PHRASE} completely {keyword} to me.",
    "I'm {keyword} by {SITUATION_NOUN_PHRASE}; it's just wrong.",
    "How can anyone not find {SITUATION_NOUN_PHRASE} {keyword}?",
    "It's a {keyword} display of {SITUATION_NOUN_PHRASE}, to be honest.",
    "{OBJECT_OF_EMOTION} is {keyword}. I don't even want to think about it.",
    "I am thoroughly {keyword} by {OBJECT_OF_EMOTION}.",
    "The mere thought of {OBJECT_OF_EMOTION} makes me feel {keyword}.",
    "This whole thing about {OBJECT_OF_EMOTION} is deeply {keyword}.",
    "I find {OBJECT_OF_EMOTION} to be quite {keyword}."
]

SITUATION_NOUN_PHRASES = [
    "this whole affair", "that kind of talk", "the way things are going", "people's insensitivity",
    "the recent decision", "such blatant disregard", "their actions", "this entire mess",
    "the article I just read", "his statement", "her response", "the policy"
]
ACTION_VERB_PHRASES = ["is", "feels", "seems", "appears to be", "comes across as"]
REASON_CLAUSES = [
    "because it's so disrespectful", "considering the circumstances", "it's just not right",
    "it's completely uncalled for", "they should know better", "it's a matter of principle"
]
OBJECTS_OF_EMOTION = [
    "that kind of behavior", "this level of incompetence", "the sheer audacity", "such hypocrisy",
    "the injustice of it all", "the way they handled it", "this ridiculous outcome", "their blatant lies"
]

# --- Helper Functions (Adapted from generate_fear_comments.py) ---

def generate_comment_text_single(emotion_type):
    if emotion_type == "anger":
        template = random.choice(ANGER_TEMPLATES)
        keyword = random.choice(ANGER_KEYWORDS)
    elif emotion_type == "disgust":
        template = random.choice(DISGUST_TEMPLATES)
        keyword = random.choice(DISGUST_KEYWORDS)
    else:
        return "Error: Invalid emotion type for generation."

    comment = template.replace("{keyword}", keyword)
    if "{SITUATION_NOUN_PHRASE}" in comment:
        comment = comment.replace("{SITUATION_NOUN_PHRASE}", random.choice(SITUATION_NOUN_PHRASES))
    if "{ACTION_VERB_PHRASE}" in comment:
        comment = comment.replace("{ACTION_VERB_PHRASE}", random.choice(ACTION_VERB_PHRASES))
    if "{REASON_CLAUSE}" in comment:
        comment = comment.replace("{REASON_CLAUSE}", random.choice(REASON_CLAUSES))
    if "{OBJECT_OF_EMOTION}" in comment:
        comment = comment.replace("{OBJECT_OF_EMOTION}", random.choice(OBJECTS_OF_EMOTION))
    
    # Basic fallback for any missed placeholders
    for placeholder_list in [SITUATION_NOUN_PHRASES, ACTION_VERB_PHRASES, REASON_CLAUSES, OBJECTS_OF_EMOTION]:
        for placeholder_text in placeholder_list: # Not the placeholder itself, but an example from the list
             # This is a simplification; ideally, ensure templates are robust
            if "{" in comment and "}" in comment:
                 start_idx = comment.find("{")
                 end_idx = comment.find("}")
                 if start_idx != -1 and end_idx != -1:
                      comment = comment.replace(comment[start_idx : end_idx+1], random.choice(placeholder_list))
                 else: # no more placeholders
                      break 
    return comment.capitalize()

def load_used_comment_ids(file_paths):
    used_ids = set()
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, usecols=['comment_id'])
            used_ids.update(df['comment_id'].tolist())
            print(f"Successfully loaded {len(df['comment_id'])} IDs from {file_path}")
        except FileNotFoundError:
            print(f"Warning: Manually labelled file not found (for IDs): {file_path}")
        except Exception as e:
            print(f"Error reading {file_path} for IDs: {e}")
    return used_ids

def get_unused_comment_metadata(no_spam_file_paths, used_comment_ids, num_needed, all_original_columns):
    unused_metadata = []
    # Define core metadata columns that should be prioritized if available.
    # Ensure these names match exactly with columns in your no_spam_files.
    core_metadata_cols = ['comment_id', 'author_name', 'published_at', 'published_at_unix', 'like_count']
    
    # Determine which of the core metadata columns are actually present in all_original_columns.
    # Also include 'detected_language' for filtering, and 'original_comment' just in case (though it's replaced).
    cols_to_attempt_loading = [col for col in core_metadata_cols if col in all_original_columns]
    cols_to_attempt_loading.extend(['detected_language', 'original_comment'])
    cols_to_attempt_loading = list(set(cols_to_attempt_loading)) # Ensure unique

    print(f"Attempting to load columns for metadata: {cols_to_attempt_loading}")
    
    # Use all_original_columns to define the full structure for each metadata dict.
    # This ensures that the dictionaries created will have all keys expected by the final DataFrame.

    for file_path in no_spam_file_paths:
        if len(unused_metadata) >= num_needed:
            break
        try:
            print(f"Processing metadata from: {file_path}")
            chunk_iter = pd.read_csv(
                file_path,
                # Try to load only necessary columns to save memory, but handle if some are missing.
                usecols=lambda x: x in cols_to_attempt_loading or x in all_original_columns,
                chunksize=20000, # Increased chunk size
                low_memory=False
            )
            for chunk_num, chunk in enumerate(chunk_iter):
                print(f"  Processing chunk {chunk_num+1} from {file_path}...")
                # Filter for English comments not already used
                # Ensure 'comment_id' and 'detected_language' are in the chunk
                if 'comment_id' not in chunk.columns or 'detected_language' not in chunk.columns:
                    print(f"    Skipping chunk: missing 'comment_id' or 'detected_language'. Chunk columns: {chunk.columns.tolist()}")
                    continue

                potential_comments = chunk[
                    (~chunk['comment_id'].isin(used_comment_ids)) &
                    (chunk['detected_language'] == 'en')
                ]
                print(f"    Found {len(potential_comments)} potential comments in chunk.")
                
                for _, row_series in potential_comments.iterrows():
                    # Initialize meta dict with all original columns, defaulting to pd.NA
                    meta = {col: pd.NA for col in all_original_columns}
                    
                    # Populate with values from the row_series if column exists in it
                    for col in all_original_columns:
                        if col in row_series.index:
                            meta[col] = row_series[col]
                    
                    # Specific handling for core metadata to ensure they are somewhat sensible
                    meta['comment_id'] = meta.get('comment_id', f"fallbackid_{uuid.uuid4()}") # Should exist due to filter
                    meta['author_name'] = meta.get('author_name', f"AnonUser_{random.randint(1000,9999)}")
                    
                    published_at_val = meta.get('published_at')
                    try:
                        if pd.isna(published_at_val) or not isinstance(published_at_val, str): raise ValueError
                        datetime.fromisoformat(str(published_at_val).replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        fallback_dt = datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1, 365))
                        meta['published_at'] = fallback_dt.isoformat().replace('+00:00', 'Z')

                    published_at_unix_val = meta.get('published_at_unix')
                    if pd.isna(published_at_unix_val):
                        try:
                            meta['published_at_unix'] = int(datetime.fromisoformat(str(meta['published_at']).replace('Z', '+00:00')).timestamp())
                        except (ValueError, TypeError):
                             meta['published_at_unix'] = int((datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1,365))).timestamp())
                    else:
                         meta['published_at_unix'] = int(published_at_unix_val)
                    
                    like_count_val = meta.get('like_count')
                    meta['like_count'] = int(like_count_val) if pd.notna(like_count_val) and isinstance(like_count_val, (int,float)) else 0

                    unused_metadata.append(meta)
                    if len(unused_metadata) >= num_needed:
                        break
                if len(unused_metadata) >= num_needed:
                    break
        except FileNotFoundError:
            print(f"Warning: No-spam source file not found: {file_path}")
        except Exception as e:
            print(f"Error reading or processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    random.shuffle(unused_metadata)
    print(f"Finished collecting metadata. Found {len(unused_metadata)} unused entries.")
    return unused_metadata


if __name__ == "__main__":
    print("Starting script...")
    # Determine column structure from one of the manually labelled files (could be the first one)
    # This file must exist and be readable.
    structure_file_path = MANUALLY_LABELLED_FILES_FOR_IDS[0]
    try:
        print(f"Reading column structure from: {structure_file_path}")
        df_structure = pd.read_csv(structure_file_path)
        original_columns = df_structure.columns.tolist()
        print(f"Successfully read column structure. Columns: {original_columns}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read '{structure_file_path}' to determine column structure: {e}")
        print("Using a hardcoded fallback column structure. THIS MAY CAUSE ISSUES if it doesn't match your data.")
        original_columns = [
            'comment_id', 'original_comment', 'comment_no_emojis', 'emojis_in_comment',
            'emoji_text_representation', 'detected_language', 'author_name', 'published_at',
            'published_at_unix', 'like_count', 'emotion_anger', 'emotion_disgust',
            'emotion_fear', 'emotion_joy', 'emotion_neutral', 'emotion_sadness',
            'emotion_surprise', 'manual_emotion_label_mb', 'manual_emotion_label_eden'
        ]
    
    used_ids = load_used_comment_ids(MANUALLY_LABELLED_FILES_FOR_IDS)
    print(f"Total loaded used comment IDs: {len(used_ids)}.")

    num_total_to_generate = NUM_PER_EMOTION * 2 # For anger and disgust
    
    # Fetch more metadata than strictly needed, in case some are unsuitable or to have variety
    available_metadata = get_unused_comment_metadata(NO_SPAM_FILES_FOR_METADATA, used_ids, num_total_to_generate + 50, original_columns)
    
    if len(available_metadata) < num_total_to_generate:
        print(f"Warning: Found only {len(available_metadata)} unique metadata entries, but need {num_total_to_generate}.")
        print("Will proceed, but some comments might get fallback/less ideal metadata or generation might be short.")

    all_generated_comments_data = []
    metadata_idx_counter = 0

    for emotion_label, num_gen in [("anger", NUM_PER_EMOTION), ("disgust", NUM_PER_EMOTION)]:
        print(f"Generating {num_gen} comments for emotion: {emotion_label}")
        for i in range(num_gen):
            if metadata_idx_counter >= len(available_metadata):
                print(f"  Ran out of unique real metadata at comment {i+1} for {emotion_label}. Using fallback metadata generation.")
                # Simplified fallback if no real metadata is left
                current_meta = {col: pd.NA for col in original_columns} # Initialize with NAs
                current_meta['comment_id'] = f"fallback_{emotion_label}_{uuid.uuid4()}"
                current_meta['author_name'] = f"FallbackUser{random.randint(1000,9999)}"
                fallback_dt = datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1, 365))
                current_meta['published_at'] = fallback_dt.isoformat().replace('+00:00', 'Z')
                current_meta['published_at_unix'] = int(fallback_dt.timestamp())
                current_meta['like_count'] = 0
            else:
                current_meta = available_metadata[metadata_idx_counter]
                metadata_idx_counter += 1

            generated_text = generate_comment_text_single(emotion_label)
            
            # Start with a copy of the real/fallback metadata
            row = current_meta.copy() 
            
            # Overwrite with generated content and fixed emotion data
            row['original_comment'] = generated_text
            row['comment_no_emojis'] = generated_text # Assuming generated text has no emojis
            row['emojis_in_comment'] = ''
            row['emoji_text_representation'] = ''
            row['detected_language'] = 'en'
            
            row['manual_emotion_label_mb'] = emotion_label
            row['manual_emotion_label_eden'] = emotion_label # Key label for training
            
            # Set emotion scores
            for col_emotion in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']:
                row[f'emotion_{col_emotion}'] = 1.0 if col_emotion == emotion_label else 0.0

            # Final check for any missing columns from original_columns (should be rare if metadata sourced well)
            for col in original_columns:
                if col not in row or pd.isna(row[col]): # Also fill if it was NA from source
                    if col == 'comment_id' and ('comment_id' not in row or pd.isna(row['comment_id'])) : row[col] = f"final_fallback_{emotion_label}_{uuid.uuid4()}"
                    elif col == 'author_name' and ('author_name' not in row or pd.isna(row['author_name'])): row[col] = "GeneratedCommenter"
                    # Add more specific fallbacks if needed, otherwise they remain NA or get generic value
                    elif col.startswith('emotion_') and pd.isna(row.get(col)): row[col] = 0.0 
                    elif col.startswith('manual_emotion_label_') and pd.isna(row.get(col)): row[col] = emotion_label
                    elif pd.isna(row.get(col)): # Generic NA for other missing ones
                        if 'count' in col or 'unix' in col : row[col] = 0 # for numeric
                        else: row[col] = '' # for string like

            all_generated_comments_data.append(row)

    if not all_generated_comments_data:
        print("No comments were generated. Exiting.")
    else:
        output_df = pd.DataFrame(all_generated_comments_data, columns=original_columns)
        
        # Data type consistency
        output_df['like_count'] = output_df['like_count'].fillna(0).astype(int)
        for col in ['emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy', 'emotion_neutral', 'emotion_sadness', 'emotion_surprise']:
            if col in output_df.columns:
                output_df[col] = output_df[col].fillna(0.0).astype(float)
        
        # Ensure boolean/object columns are strings if they are labels or text
        for col in ['manual_emotion_label_mb', 'manual_emotion_label_eden', 'detected_language', 'emojis_in_comment', 'emoji_text_representation']:
             if col in output_df.columns:
                output_df[col] = output_df[col].fillna('').astype(str)


        print(f"Saving {len(output_df)} generated comments to {OUTPUT_FILENAME}...")
        output_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Successfully saved. Columns: {output_df.columns.tolist()}")
        
        if metadata_idx_counter < num_total_to_generate and len(available_metadata) < num_total_to_generate :
             print(f"Note: Only {metadata_idx_counter} comments received unique real metadata due to limited availability ({len(available_metadata)} found).")
        elif len(available_metadata) >= num_total_to_generate:
             print(f"Successfully used {num_total_to_generate} unique real metadata entries.")


    print("Script finished.") 