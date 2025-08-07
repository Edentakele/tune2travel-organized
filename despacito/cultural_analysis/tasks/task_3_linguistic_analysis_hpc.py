import pandas as pd
import re
import math
import os
import time
import multiprocessing # Added for parallel processing

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    print("WARNING: language-tool-python not found. Grammar checking will be skipped. Install with: pip install language-tool-python")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("WARNING: textstat not found. Readability scores will be skipped. Install with: pip install textstat")

# This cache will be process-local when using multiprocessing
LANGUAGE_TOOLS_CACHE = {}

def get_language_tool(lang_code):
    if not LANGUAGE_TOOL_AVAILABLE:
        return None
    if lang_code in LANGUAGE_TOOLS_CACHE:
        return LANGUAGE_TOOLS_CACHE[lang_code]
    try:
        # Basic normalization/mapping if specific language codes are problematic for LanguageTool
        # e.g., if LT expects 'pt-BR' but gets 'pt', though it's often flexible.
        # lang_code = lang_code.replace('-', '_') # Example: some tools prefer en_US over en-US
        tool = language_tool_python.LanguageTool(lang_code)
        LANGUAGE_TOOLS_CACHE[lang_code] = tool
        # print(f"Process {os.getpid()}: Initialized LanguageTool for: {lang_code}") # Verbose, disable for cleaner output
        return tool
    except Exception as e:
        # print(f"Process {os.getpid()}: Could not initialize LT for '{lang_code}'. Error: {e}.") # Verbose
        LANGUAGE_TOOLS_CACHE[lang_code] = None
        return None

def get_grammar_error_count_for_worker(text, lang_code):
    if not LANGUAGE_TOOL_AVAILABLE or not text or pd.isna(text):
        return 0
    tool = get_language_tool(lang_code)
    if tool:
        try:
            return len(tool.check(str(text)))
        except Exception:
            # print(f"Process {os.getpid()}: LT.check error for lang {lang_code}") # Verbose
            return 0
    return 0

def get_readability_score_flesch_for_worker(text):
    if not TEXTSTAT_AVAILABLE or not text or pd.isna(text):
        return 0
    try:
        return textstat.flesch_reading_ease(str(text))
    except Exception:
        return 0

def worker_process_comment_features(task_data):
    """Processes a single comment to get grammar errors and readability.
    Args:
        task_data: A tuple (original_df_index, comment_text, lang_code)
    Returns:
        A tuple (original_df_index, grammar_error_count, readability_flesch_score)
    """
    original_df_index, comment_text, lang_code = task_data
    # Ensure lang_code is a string, as it might come from pandas as non-string type if it was NaN before fillna
    lang_code_str = str(lang_code if pd.notna(lang_code) else 'en')
    
    grammar_errors = get_grammar_error_count_for_worker(comment_text, lang_code_str)
    readability_score = get_readability_score_flesch_for_worker(comment_text)
    return original_df_index, grammar_errors, readability_score

def load_single_file_data(file_path):
    try:
        df = pd.read_csv(file_path, usecols=['comment_id', 'original_comment', 'detected_language'])
        df.dropna(subset=['original_comment'], inplace=True)
        df['original_comment'] = df['original_comment'].astype(str)
        if 'comment_id' in df.columns:
            df['comment_id'] = df['comment_id'].astype(str)
        
        # This column is used by the worker for LanguageTool
        if 'detected_language' in df.columns:
            df['detected_language_for_tool'] = df['detected_language'].fillna('en').astype(str)
        else:
            df['detected_language_for_tool'] = 'en' # Default if column is entirely missing
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Check columns: 'comment_id', 'original_comment', 'detected_language'.")
        return pd.DataFrame()

def get_words(text):
    if not isinstance(text, str): return []
    words = text.lower().split()
    words = [re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word) for word in words]
    return [word for word in words if word]

def average_word_length(text):
    words = get_words(text)
    if not words: return 0
    return sum(len(word) for word in words) / len(words)

def lexical_diversity(text):
    words = get_words(text)
    if not words: return 0
    return len(set(words)) / len(words)

def uppercase_proportion(text):
    if not isinstance(text, str) or not text: return 0
    return sum(1 for char in text if char.isupper()) / len(text)

def count_repeated_characters_sequences(text):
    if not isinstance(text, str): return 0
    return len(re.findall(r'(.)\1{2,}', text))

def calculate_enhanced_language_score(row):
    score = 0
    # Use .get(column, default_value) for robustness if a column might be missing (e.g. if a feature calc failed)
    readability_normalized = (row.get('readability_flesch', 0) - 50) / 50
    score += readability_normalized * 0.4
    grammar_penalty = math.log1p(row.get('grammar_error_count', 0)) * 0.3
    score -= grammar_penalty
    score += row.get('lexical_diversity', 0) * 0.15
    avg_word_len_normalized = (row.get('avg_word_length', 0) - 4) / 4
    score += avg_word_len_normalized * 0.05
    score -= row.get('uppercase_proportion', 0) * 0.05
    score -= math.log1p(row.get('repeated_chars_count', 0)) * 0.05
    return score

def get_file_prefix(file_path):
    base_name = os.path.basename(file_path)
    if "despa_" in base_name: return "despa_"
    elif "seeyou_" in base_name: return "seeyou_"
    return "unknown_"

def print_task3_notes():
    print("\n--- Important Notes for Enhanced Task 3 ---")
    print("1. The 'enhanced_language_score' incorporates grammar errors and readability.")
    print("   It remains heuristic but is more robust than the previous simple score.")
    print("2. Ensure Java is installed (JDK 17+ recommended) and configured for 'language-tool-python'.")
    print("3. Processing time has been improved with parallel processing for grammar/readability checks.")
    print("4. Review comments with very high/low scores, or high error counts / low readability.")

def main():
    file_paths = [
        '../final_extraction/despa_kJQP7kiw5Fk_comments_spam_emotion.csv',
        '../final_extraction/seeyou_RgKAFK5djSk_comments_spam_emotion.csv'
    ]

    try:
        num_cpus = os.cpu_count()
        # Leave one core free, or use a max like 6 to avoid too many LT Java instances.
        num_processes = max(1, num_cpus - 1 if num_cpus and num_cpus > 1 else 1)
        num_processes = min(num_processes, 6) 
    except NotImplementedError:
        num_processes = 2 
    print(f"INFO: Using {num_processes} processes for parallel tasks.")

    if not LANGUAGE_TOOL_AVAILABLE:
        print("REMINDER: Grammar checking is disabled (language-tool-python not found/Java issue).")
    if not TEXTSTAT_AVAILABLE:
        print("REMINDER: Readability scoring is disabled (textstat library not found).")

    for file_path in file_paths:
        prefix = get_file_prefix(file_path)
        print(f"\n--- Processing file: {file_path} with prefix: {prefix} for Enhanced Task 3 ---")

        print("Loading data...")
        df_single = load_single_file_data(file_path)

        if df_single.empty:
            print(f"No data loaded for {file_path}. Skipping.")
            continue

        print(f"Loaded {len(df_single)} comments from {file_path}.")
        total_rows = len(df_single)
        progress_interval = max(1, min(1000, total_rows // 20 if total_rows > 0 else 1))
        
        print("Calculating enhanced linguistic features (grammar & readability in parallel)...")
        start_time_features = time.time()

        # Initialize result columns before parallel processing
        df_single['grammar_error_count'] = 0
        df_single['readability_flesch'] = 0.0 # float for readability

        if (LANGUAGE_TOOL_AVAILABLE or TEXTSTAT_AVAILABLE) and total_rows > 0 and num_processes > 0:
            tasks_for_pool = []
            for i in range(total_rows):
                # Pass the actual DataFrame index, comment string, and language string for the tool
                tasks_for_pool.append((
                    df_single.index[i], 
                    df_single['original_comment'].iloc[i], 
                    df_single['detected_language_for_tool'].iloc[i]
                ))
            
            # Dynamic chunksize for imap_unordered
            # Aim for enough chunks to get good progress updates without too much overhead.
            # Roughly num_processes * 4 to num_processes * 10 chunks overall.
            if total_rows > num_processes:
                chunksize = max(1, total_rows // (num_processes * 8)) 
                chunksize = min(chunksize, 250) # Cap chunksize
            else:
                chunksize = 1 # Small number of rows, small chunks
            if chunksize == 0 : chunksize = 1 # Ensure chunksize is at least 1
            
            print(f"  Starting parallel processing for {total_rows} comments with {num_processes} processes (chunksize ~{chunksize})...")
            processed_count = 0
            # Using a try-finally to ensure pool is closed even if errors occur mid-processing
            pool = multiprocessing.Pool(processes=num_processes)
            try:
                for original_idx, grammar_count, readability_score in pool.imap_unordered(worker_process_comment_features, tasks_for_pool, chunksize=chunksize):
                    df_single.loc[original_idx, 'grammar_error_count'] = grammar_count
                    df_single.loc[original_idx, 'readability_flesch'] = readability_score
                    processed_count += 1
                    if processed_count % progress_interval == 0 or processed_count == total_rows:
                        print(f"    Linguistic features (parallel) progress: {processed_count}/{total_rows} comments analyzed.")
            finally:
                pool.close()
                pool.join()
            print("  Parallel grammar/readability processing complete.")
        else:
            print("  Skipping parallel grammar/readability (libs not available, no data, or num_processes=0).")
            
        print("Calculating simple text features (avg_word_length, lexical_diversity, etc.)...")
        df_single['avg_word_length'] = df_single['original_comment'].apply(average_word_length)
        df_single['lexical_diversity'] = df_single['original_comment'].apply(lexical_diversity)
        df_single['uppercase_proportion'] = df_single['original_comment'].apply(uppercase_proportion)
        df_single['repeated_chars_count'] = df_single['original_comment'].apply(count_repeated_characters_sequences)
        
        print("Calculating final enhanced language scores...")
        df_single['enhanced_language_score'] = df_single.apply(calculate_enhanced_language_score, axis=1)

        processing_time_features = time.time() - start_time_features
        print(f"Total feature calculation for {total_rows} comments took {processing_time_features:.2f} seconds.")

        # Prepare final output columns list
        columns_for_export = []
        if 'comment_id' in df_single.columns: columns_for_export.append('comment_id')
        if 'original_comment' in df_single.columns: columns_for_export.append('original_comment')
        
        # Use the original 'detected_language' if it exists, otherwise the one created for the tool
        if 'detected_language' in df_single.columns and 'detected_language' not in columns_for_export:
            columns_for_export.append('detected_language')
        elif 'detected_language_for_tool' in df_single.columns and 'detected_language' not in columns_for_export:
            # If original 'detected_language' was missing, 'detected_language_for_tool' has the default.
            # Rename it for the output to be consistent.
            if 'detected_language' not in df_single.columns: # ensure we only rename if original is truly missing
                 df_single.rename(columns={'detected_language_for_tool': 'detected_language'}, inplace=True)
            if 'detected_language' not in columns_for_export : columns_for_export.append('detected_language')
        
        # Add analysis columns if they exist in the DataFrame
        for col_name in ['grammar_error_count', 'readability_flesch', 'avg_word_length', 'lexical_diversity', 
                         'uppercase_proportion', 'repeated_chars_count', 'enhanced_language_score']:
            if col_name in df_single.columns and col_name not in columns_for_export:
                columns_for_export.append(col_name)
        
        # Ensure no duplicate columns if somehow added and select only existing ones
        seen_cols = set()
        final_columns_for_export_ordered = []
        for col in columns_for_export:
            if col not in seen_cols and col in df_single.columns:
                final_columns_for_export_ordered.append(col)
                seen_cols.add(col)

        df_output_final = df_single[final_columns_for_export_ordered]

        output_file = f"{prefix}task_3_enhanced_linguistic_analysis.csv"
        df_output_final.to_csv(output_file, index=False)
        print(f"\nEnhanced linguistic analysis for {prefix[:-1]} saved to {output_file}")
        print(f"\nTask 3 (Enhanced & Parallelized) processing complete for {file_path}.")

    if LANGUAGE_TOOL_AVAILABLE: # Clear cache only if it was used by main process (though workers have their own)
        LANGUAGE_TOOLS_CACHE.clear()
    print("\n\nAll files processed for Enhanced Task 3.")
    print_task3_notes()

if __name__ == '__main__':
    # This check is important for multiprocessing on Windows, and good practice elsewhere
    multiprocessing.freeze_support()
    main()