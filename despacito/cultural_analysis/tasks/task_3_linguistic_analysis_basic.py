import pandas as pd
import re
import math
import os

def load_single_file_data(file_path):
    """Loads a single CSV file, keeping specified columns."""
    try:
        df = pd.read_csv(file_path, usecols=['comment_id', 'original_comment'])
        df.dropna(subset=['original_comment'], inplace=True)
        df['original_comment'] = df['original_comment'].astype(str)
        if 'comment_id' in df.columns:
            df['comment_id'] = df['comment_id'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Check if 'comment_id' and 'original_comment' columns exist.")
        return pd.DataFrame()

def get_words(text):
    if not isinstance(text, str):
        return []
    words = text.lower().split()
    words = [re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word) for word in words]
    words = [word for word in words if word]
    return words

def average_word_length(text):
    words = get_words(text)
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

def lexical_diversity(text):
    words = get_words(text)
    if not words:
        return 0
    return len(set(words)) / len(words)

def uppercase_proportion(text):
    if not isinstance(text, str) or not text:
        return 0
    return sum(1 for char in text if char.isupper()) / len(text)

def count_repeated_characters_sequences(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'(.)\1{2,}', text))

def calculate_rough_language_score(row):
    """Calculates the original heuristic language score."""
    score = 0
    score += row.get('avg_word_length', 0) * 0.2 
    score += row.get('lexical_diversity', 0) * 0.4
    score -= row.get('uppercase_proportion', 0) * 0.2 
    score -= math.log1p(row.get('repeated_chars_count', 0)) * 0.2
    return score

def get_file_prefix(file_path):
    base_name = os.path.basename(file_path)
    if "despa_" in base_name: return "despa_"
    elif "seeyou_" in base_name: return "seeyou_"
    return "unknown_"

def main():
    file_paths = [
        '../final_extraction/despa_kJQP7kiw5Fk_comments_spam_emotion.csv',
        '../final_extraction/seeyou_RgKAFK5djSk_comments_spam_emotion.csv'
    ]
    
    for file_path in file_paths:
        prefix = get_file_prefix(file_path)
        print(f"\n--- Processing file: {file_path} with prefix: {prefix} for Basic Task 3 ---")

        print("Loading data...")
        df_single = load_single_file_data(file_path)

        if df_single.empty:
            print(f"No data loaded for {file_path} or file is empty. Skipping.")
            continue

        print(f"Loaded {len(df_single)} comments from {file_path}.")

        print("Calculating basic linguistic features...")
        df_single['avg_word_length'] = df_single['original_comment'].apply(average_word_length)
        df_single['lexical_diversity'] = df_single['original_comment'].apply(lexical_diversity)
        df_single['uppercase_proportion'] = df_single['original_comment'].apply(uppercase_proportion)
        df_single['repeated_chars_count'] = df_single['original_comment'].apply(count_repeated_characters_sequences)
        
        print("Calculating rough language scores...")
        df_single['rough_language_score'] = df_single.apply(calculate_rough_language_score, axis=1)

        # Prepare final output columns list
        columns_for_export = []
        if 'comment_id' in df_single.columns: columns_for_export.append('comment_id')
        if 'original_comment' in df_single.columns: columns_for_export.append('original_comment')
        
        for col_name in ['avg_word_length', 'lexical_diversity', 'uppercase_proportion', 
                         'repeated_chars_count', 'rough_language_score']:
            if col_name in df_single.columns and col_name not in columns_for_export:
                columns_for_export.append(col_name)
        
        df_output_final = df_single[columns_for_export]

        output_file = f"{prefix}task_3_basic_linguistic_features.csv"
        df_output_final.to_csv(output_file, index=False)
        print(f"\nBasic linguistic features for {prefix[:-1]} saved to {output_file}")
        print(f"\nBasic Task 3 processing complete for {file_path}.")

    print("\n\nAll files processed for Basic Task 3.")
    print("\n--- Notes for Basic Task 3 ---")
    print("1. This script calculates a 'rough_language_score' based on simple text features.")
    print("2. It does NOT perform grammatical error checking or deep fluency analysis.")
    print("3. For more robust analysis, use the 'task_3_enhanced_linguistic_analysis.py' script (requires Java & additional libraries).")

if __name__ == '__main__':
    main() 