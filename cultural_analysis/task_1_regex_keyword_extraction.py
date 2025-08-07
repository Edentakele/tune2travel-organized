import pandas as pd
import re
import os # Added for filename manipulation
from collections import Counter

# Define keywords and patterns
KEYWORDS = [
    "hi from",
    "love from",
    "greetings from",
    "watching from"
]

# Regex for flag emojis (Unicode range for most flags)
# This is a simplified range and might not capture all flag emojis.
# Regional Indicator Symbols (U+1F1E6 to U+1F1FF) are used in pairs.
FLAG_EMOJI_PATTERN = re.compile(r"[🇦-🇿]{2}")

# Placeholder for country/city name extraction
# A robust solution would require a comprehensive list of countries/cities
# or a library like geograpy3, pycountry, or spaCy's NER.
# For now, we'll use a very simple regex for demonstration.
LOCATION_PATTERN = re.compile(
    r"\b(India|Turkey|USA|UK|Canada|Australia|Germany|France|Brazil|Japan|China|Russia|Italy|Spain|Mexico|Argentina|Egypt|Nigeria|Kenya|Pakistan|Bangladesh|[A-Za-z]+[Cc]ity)\b",
    re.IGNORECASE
)

def load_data_with_spam_info(file_path):
    """Loads a single CSV file with comment, date, and spam information, handling different spam column names."""
    try:
        # Read header to find available columns
        try:
            df_cols = pd.read_csv(file_path, nrows=0, low_memory=False).columns.tolist()
        except pd.errors.EmptyDataError:
            print(f"Warning: File is empty, skipping: {file_path}")
            return pd.DataFrame()

        spam_col_name = None
        if 'spam_label' in df_cols:
            spam_col_name = 'spam_label'
        elif 'is_spam' in df_cols:
            spam_col_name = 'is_spam'
        
        required_cols = ['original_comment', 'published_at']
        if spam_col_name:
            required_cols.append(spam_col_name)

        if 'emotion_label' in df_cols:
            required_cols.append('emotion_label')

        df = pd.read_csv(file_path, usecols=required_cols, low_memory=False)

        if spam_col_name and spam_col_name != 'spam_label':
            df.rename(columns={spam_col_name: 'spam_label'}, inplace=True)

        if 'spam_label' not in df.columns:
            print(f"Warning: No spam column ('spam_label' or 'is_spam') found in {file_path}. Adding a default column.")
            df['spam_label'] = 'unknown'
        
        if 'emotion_label' not in df.columns:
            print(f"Warning: No emotion column ('emotion_label') found in {file_path}. Adding a default column.")
            df['emotion_label'] = 'unknown'

        df.dropna(subset=['original_comment', 'published_at'], inplace=True)
        df['original_comment'] = df['original_comment'].astype(str)
        
        if 'spam_label' in df.columns:
            df['spam_label'] = df['spam_label'].fillna('unknown')
        
        if 'emotion_label' in df.columns:
            df['emotion_label'] = df['emotion_label'].fillna('unknown')

        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def extract_comments_by_keyword(df, comment_column='original_comment'):
    """Extracts comments containing specified keywords."""
    pattern = "|".join([re.escape(keyword) for keyword in KEYWORDS])
    return df[df[comment_column].str.contains(pattern, case=False, na=False)]

def extract_comments_by_flag_emoji(df, comment_column='original_comment'):
    """Extracts comments containing flag emojis."""
    return df[df[comment_column].str.contains(FLAG_EMOJI_PATTERN, na=False)]

def extract_comments_by_location(df, comment_column='original_comment'):
    """Extracts comments mentioning predefined locations (basic)."""
    # Note: This is a very basic location extraction.
    return df[df[comment_column].str.contains(LOCATION_PATTERN, na=False)]

def analyze_temporal_distribution(df, timestamp_column='published_at'):
    """Analyzes the temporal distribution of comments."""
    if df.empty or timestamp_column not in df.columns:
        return pd.Series(dtype='int64')
    
    df_copy = df.copy() # Create a copy to avoid SettingWithCopyWarning
    try:
        df_copy['timestamp'] = pd.to_datetime(df_copy[timestamp_column], errors='coerce')
        df_copy.dropna(subset=['timestamp'], inplace=True) # Drop rows where conversion failed
        df_copy['year_month'] = df_copy['timestamp'].dt.to_period('M')
        temporal_distribution = df_copy['year_month'].value_counts().sort_index()
        return temporal_distribution
    except Exception as e:
        print(f"Error during temporal analysis for a file: {e}")
        return pd.Series(dtype='int64')

def get_file_prefix(file_path):
    """Generates a prefix from the filename."""
    base_name = os.path.basename(file_path)
    if "despa_" in base_name:
        return "despa_"
    elif "seeyou_" in base_name:
        return "seeyou_"
    return "unknown_"

def main():
    # Corrected paths relative to the script's location in 'cultural_analysis'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_paths = [
        os.path.join(script_dir, '../final_extraction/despa_kJQP7kiw5Fk_comments_spam_emotion.csv'),
        os.path.join(script_dir, '../final_extraction/seeyou_RgKAFK5djSk_comments_spam_emotion.csv')
    ]
    
    for file_path in file_paths:
        prefix = get_file_prefix(file_path)
        print(f"\n--- Processing file: {file_path} with prefix: {prefix} ---")
        
        print("Loading data with spam information...")
        df_single = load_data_with_spam_info(file_path)

        if df_single.empty:
            print(f"No data loaded for {file_path} or file is empty. Skipping.")
            continue

        print(f"Loaded {len(df_single)} comments from {file_path}.")

        # Define columns to save, ensuring spam_label is included
        columns_to_save = ['original_comment', 'published_at', 'spam_label', 'emotion_label']

        print("\nExtracting comments with keywords...")
        df_keyword_comments = extract_comments_by_keyword(df_single)
        print(f"Found {len(df_keyword_comments)} comments with keywords.")
        df_keyword_comments[columns_to_save].to_csv(f"{prefix}task_1_keyword_extracted_comments.csv", index=False)

        print("\nExtracting comments with flag emojis...")
        df_flag_emoji_comments = extract_comments_by_flag_emoji(df_single)
        print(f"Found {len(df_flag_emoji_comments)} comments with flag emojis.")
        df_flag_emoji_comments[columns_to_save].to_csv(f"{prefix}task_1_flag_emoji_extracted_comments.csv", index=False)

        print("\nExtracting comments mentioning locations (basic)...")
        df_location_comments = extract_comments_by_location(df_single)
        print(f"Found {len(df_location_comments)} comments mentioning locations.")
        df_location_comments[columns_to_save].to_csv(f"{prefix}task_1_location_extracted_comments.csv", index=False)

        # Combine all extracted comments for this file for its overall temporal analysis
        df_all_extracted_for_file = pd.concat([
            df_keyword_comments, 
            df_flag_emoji_comments, 
            df_location_comments
        ]).drop_duplicates().reset_index(drop=True)
        
        print(f"\nTotal unique comments extracted for {prefix[:-1]}: {len(df_all_extracted_for_file)}")
        
        # Ensure emotion_label column is present before saving
        if 'emotion_label' not in df_all_extracted_for_file.columns:
            df_all_extracted_for_file['emotion_label'] = 'unknown'

        df_all_extracted_for_file[columns_to_save].to_csv(f"{prefix}task_1_all_extracted_comments.csv", index=False)

        print("\nAnalyzing temporal distribution of all extracted comments for this file...")
        temporal_dist = analyze_temporal_distribution(df_all_extracted_for_file)
        if not temporal_dist.empty:
            print("Temporal Distribution (Year-Month):")
            print(temporal_dist)
            temporal_dist.to_csv(f"{prefix}task_1_temporal_distribution.csv")
        else:
            print(f"Could not generate temporal distribution for {prefix[:-1]} (no relevant data or 'published_at' column missing/invalid).")

        print(f"\nTask 1 processing complete for {file_path}. Results saved with prefix '{prefix}'.")

    print("\n\nAll files processed for Task 1.")
    print("Note: Location extraction is basic. For more accurate results, consider using a dedicated NER/geocoding tool.")

if __name__ == '__main__':
    main() 