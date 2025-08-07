import pandas as pd
import re
import os # Added for filename manipulation
import csv # To handle special characters in CSV writing
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas(desc="Labeling Comments")

# Define keyword categories as per the new requirements
KEYWORD_CATEGORIES = {
    'Ethnic Identity': ['latino', 'hispanic', 'asian', 'white', 'caucasian', 'black'],
    'Religious Identity': ['muslim', 'islam', 'christian', 'christ'],
    'Collective Memory': ['my country', 'our country', 'in my culture', 'homeland', 'diaspora', 'immigrant']
}

# Combine all keywords for the initial regex filter
ALL_KEYWORDS = [keyword for sublist in KEYWORD_CATEGORIES.values() for keyword in sublist]

def load_data_with_date(file_path):
    """Loads a CSV file, keeping only the comment and its publication date."""
    try:
        df = pd.read_csv(file_path, usecols=['original_comment', 'published_at'])
        df.dropna(subset=['original_comment', 'published_at'], inplace=True)
        # Sanitize comment text to prevent CSV corruption from multiline comments
        df['original_comment'] = df['original_comment'].astype(str).str.replace(r'\r|\n', ' ', regex=True)
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df.dropna(subset=['published_at'], inplace=True) # Drop rows where date conversion failed
        df['Year'] = df['published_at'].dt.year
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def filter_and_label_comments(df):
    """Filters comments by keywords and assigns keywords to separate context columns."""
    # Create a regex pattern to find any of the keywords for initial filtering
    regex_pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in ALL_KEYWORDS) + r')\b'
    
    # Filter rows that contain at least one of the keywords
    filtered_df = df[df['original_comment'].str.contains(regex_pattern, case=False, na=False)].copy()
    
    if filtered_df.empty:
        return pd.DataFrame()

    # --- Assign keywords to their respective category columns ---
    def find_keywords_in_categories(comment):
        """Checks a comment for keywords from each category."""
        comment_lower = comment.lower()
        # Find the first matched keyword for each category
        found_keywords = {
            label: next((keyword for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', comment_lower)), None)
            for label, keywords in KEYWORD_CATEGORIES.items()
        }
        return found_keywords

    # Apply the function to the filtered dataframe, resulting in a Series of dictionaries
    keyword_dicts = filtered_df['original_comment'].progress_apply(find_keywords_in_categories)
    
    # Convert the Series of dicts into a new DataFrame
    labels_df = pd.json_normalize(keyword_dicts)
    labels_df.index = filtered_df.index
    
    # Join the new label columns to the original filtered dataframe
    result_df = filtered_df.join(labels_df)
    
    # Define the final columns order
    final_columns = ['original_comment', 'Year'] + list(KEYWORD_CATEGORIES.keys())
    
    return result_df[final_columns]

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
        print(f"\n--- Processing file: {file_path} with prefix: {prefix} for Task 2 ---")

        print("Loading data with dates...")
        df_full = load_data_with_date(file_path)

        if df_full.empty:
            print(f"No data loaded for {file_path} or file is empty. Skipping.")
            continue

        print(f"Loaded {len(df_full)} comments with valid dates from {file_path}.")

        print("\nFiltering comments by context keywords and assigning labels...")
        df_labeled_comments = filter_and_label_comments(df_full)
        
        if df_labeled_comments.empty:
            print(f"Found 0 comments based on context keywords for {prefix[:-1]}.")
            continue

        print(f"Found and labeled {len(df_labeled_comments)} comments for {prefix[:-1]}.")
        
        # --- Split by year and save to subdirectory ---
        output_dir = os.path.join(script_dir, f"{prefix}task_2_yearly_split")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory for yearly outputs: {output_dir}")

        # Get unique years and sort them
        years = sorted(df_labeled_comments['Year'].unique())

        for year in years:
            df_year = df_labeled_comments[df_labeled_comments['Year'] == year]
            
            # Define output filename for the year
            output_filename = os.path.join(output_dir, f"{prefix}{year}_labeled_context_comments.csv")
            
            print(f"Saving {len(df_year)} comments for year {year} to {output_filename}")
            
            # Use QUOTE_ALL to handle multiline comments correctly
            df_year.to_csv(output_filename, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')

        print(f"\nTask 2 processing complete for {file_path}. Yearly files saved in '{output_dir}'.")

    print("\n\nAll files processed for Task 2.")

if __name__ == '__main__':
    main() 