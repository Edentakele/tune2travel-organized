import pandas as pd
import re
import os
from collections import Counter

# Spanish-specific keywords for cultural analysis
SPANISH_KEYWORDS = [
    "saludos desde",
    "amor desde", 
    "viendo desde",
    "escribiendo desde",
    "soy de",
    "vivo en",
    "desde",
    "aquí en",
    "mi país",
    "mi ciudad",
    "mi cultura",
    "nosotros los",
    "somos",
    "patria",
    "homeland",
    "raíces"
]

# Spanish cultural identity keywords
SPANISH_CULTURAL_KEYWORDS = [
    "latino", "latina", "hispanico", "hispana",
    "mexicano", "mexicana", "colombiano", "colombiana",
    "argentino", "argentina", "español", "española",
    "puertorriqueño", "puertorriqueña", "dominicano", "dominicana",
    "venezolano", "venezolana", "peruano", "peruana",
    "chileno", "chilena", "ecuatoriano", "ecuatoriana",
    "boliviano", "boliviana", "paraguayo", "paraguaya",
    "uruguayo", "uruguaya", "costarricense", "panameño", "panameña",
    "guatemalteco", "guatemalteca", "hondureño", "hondureña",
    "salvadoreño", "salvadoreña", "nicaragüense",
    "cubano", "cubana"
]

# Religious/cultural identity in Spanish
SPANISH_RELIGIOUS_KEYWORDS = [
    "católico", "católica", "cristiano", "cristiana",
    "musulmán", "musulmana", "judío", "judía",
    "protestante", "evangelico", "evangelica"
]

# Spanish location patterns (countries and major cities)
SPANISH_LOCATION_PATTERN = re.compile(
    r"\b(España|México|Argentina|Colombia|Venezuela|Perú|Chile|Ecuador|"
    r"Bolivia|Paraguay|Uruguay|Costa Rica|Panamá|Guatemala|Honduras|"
    r"El Salvador|Nicaragua|Cuba|República Dominicana|Puerto Rico|"
    r"Madrid|Barcelona|México|Bogotá|Buenos Aires|Lima|Santiago|Caracas|"
    r"Quito|La Paz|Montevideo|San José|Ciudad de Panamá|Guatemala|"
    r"Tegucigalpa|San Salvador|Managua|La Habana|Santo Domingo|San Juan)\b",
    re.IGNORECASE
)

# Flag emoji pattern (same as original)
FLAG_EMOJI_PATTERN = re.compile(r"[🇦-🇿]{2}")

def load_spanish_data(file_path):
    """Loads Spanish comments from CSV, filtering by language if available."""
    try:
        # First check what columns are available
        df_cols = pd.read_csv(file_path, nrows=0, low_memory=False).columns.tolist()
        
        required_cols = ['original_comment', 'published_at']
        optional_cols = ['detected_language', 'spam_label', 'emotion_label']
        
        # Add available optional columns
        cols_to_load = required_cols.copy()
        for col in optional_cols:
            if col in df_cols:
                cols_to_load.append(col)
        
        df = pd.read_csv(file_path, usecols=cols_to_load, low_memory=False)
        
        # Filter for Spanish comments if language detection is available
        if 'detected_language' in df.columns:
            spanish_df = df[df['detected_language'] == 'es'].copy()
            print(f"Found {len(spanish_df)} Spanish comments out of {len(df)} total comments")
        else:
            print("No language detection column found, processing all comments")
            spanish_df = df.copy()
        
        # Clean data
        spanish_df.dropna(subset=['original_comment', 'published_at'], inplace=True)
        spanish_df['original_comment'] = spanish_df['original_comment'].astype(str)
        
        # Add default columns if missing
        if 'spam_label' not in spanish_df.columns:
            spanish_df['spam_label'] = 'unknown'
        if 'emotion_label' not in spanish_df.columns:
            spanish_df['emotion_label'] = 'unknown'
            
        return spanish_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Spanish data from {file_path}: {e}")
        return pd.DataFrame()

def extract_spanish_keywords(df, comment_column='original_comment'):
    """Extracts Spanish comments containing cultural keywords."""
    pattern = "|".join([re.escape(keyword) for keyword in SPANISH_KEYWORDS])
    return df[df[comment_column].str.contains(pattern, case=False, na=False)]

def extract_spanish_cultural_identity(df, comment_column='original_comment'):
    """Extracts Spanish comments with cultural identity markers."""
    pattern = "|".join([re.escape(keyword) for keyword in SPANISH_CULTURAL_KEYWORDS])
    return df[df[comment_column].str.contains(pattern, case=False, na=False)]

def extract_spanish_religious_identity(df, comment_column='original_comment'):
    """Extracts Spanish comments with religious identity markers."""
    pattern = "|".join([re.escape(keyword) for keyword in SPANISH_RELIGIOUS_KEYWORDS])
    return df[df[comment_column].str.contains(pattern, case=False, na=False)]

def extract_spanish_locations(df, comment_column='original_comment'):
    """Extracts Spanish comments mentioning Spanish-speaking locations."""
    return df[df[comment_column].str.contains(SPANISH_LOCATION_PATTERN, na=False)]

def extract_flag_emojis(df, comment_column='original_comment'):
    """Extracts comments containing flag emojis (same as original)."""
    return df[df[comment_column].str.contains(FLAG_EMOJI_PATTERN, na=False)]

def analyze_spanish_temporal_distribution(df, timestamp_column='published_at'):
    """Analyzes temporal distribution of Spanish comments."""
    if df.empty or timestamp_column not in df.columns:
        return pd.Series(dtype='int64')
    
    df_copy = df.copy()
    try:
        df_copy['timestamp'] = pd.to_datetime(df_copy[timestamp_column], errors='coerce')
        df_copy.dropna(subset=['timestamp'], inplace=True)
        df_copy['year_month'] = df_copy['timestamp'].dt.to_period('M')
        temporal_distribution = df_copy['year_month'].value_counts().sort_index()
        return temporal_distribution
    except Exception as e:
        print(f"Error during Spanish temporal analysis: {e}")
        return pd.Series(dtype='int64')

def get_file_prefix(file_path):
    """Generates Spanish-specific prefix from filename."""
    base_name = os.path.basename(file_path)
    if "despa_" in base_name:
        return "despa_spanish_"
    elif "seeyou_" in base_name:
        return "seeyou_spanish_"
    return "spanish_"

def main():
    # Process Spanish comments from Despacito primarily
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir  # Save outputs in the cultural_analysis_spanish folder
    file_paths = [
        os.path.join(script_dir, '../final_extraction/despa_kJQP7kiw5Fk_comments_spam_emotion.csv'),
        # Uncomment if you want to analyze Spanish comments from See You Again too
        # os.path.join(script_dir, '../final_extraction/seeyou_RgKAFK5djSk_comments_spam_emotion.csv')
    ]
    
    for file_path in file_paths:
        prefix = get_file_prefix(file_path)
        print(f"\n--- Processing Spanish comments from: {file_path} ---")
        
        # Load Spanish comments
        df_spanish = load_spanish_data(file_path)
        
        if df_spanish.empty:
            print(f"No Spanish data found in {file_path}")
            continue
            
        print(f"Processing {len(df_spanish)} Spanish comments")
        
        # Define columns to save
        columns_to_save = ['original_comment', 'published_at', 'spam_label', 'emotion_label']
        if 'detected_language' in df_spanish.columns:
            columns_to_save.append('detected_language')
        
        # Extract different types of Spanish cultural markers
        print("\n1. Extracting Spanish cultural keywords...")
        df_cultural_keywords = extract_spanish_keywords(df_spanish)
        print(f"Found {len(df_cultural_keywords)} comments with Spanish cultural keywords")
        df_cultural_keywords[columns_to_save].to_csv(os.path.join(output_dir, f"{prefix}cultural_keywords.csv"), index=False)
        
        print("\n2. Extracting Spanish cultural identity markers...")
        df_cultural_identity = extract_spanish_cultural_identity(df_spanish)
        print(f"Found {len(df_cultural_identity)} comments with cultural identity markers")
        df_cultural_identity[columns_to_save].to_csv(os.path.join(output_dir, f"{prefix}cultural_identity.csv"), index=False)
        
        print("\n3. Extracting Spanish religious identity markers...")
        df_religious_identity = extract_spanish_religious_identity(df_spanish)
        print(f"Found {len(df_religious_identity)} comments with religious identity markers")
        df_religious_identity[columns_to_save].to_csv(os.path.join(output_dir, f"{prefix}religious_identity.csv"), index=False)
        
        print("\n4. Extracting Spanish location mentions...")
        df_locations = extract_spanish_locations(df_spanish)
        print(f"Found {len(df_locations)} comments mentioning Spanish-speaking locations")
        df_locations[columns_to_save].to_csv(os.path.join(output_dir, f"{prefix}locations.csv"), index=False)
        
        print("\n5. Extracting flag emojis...")
        df_flags = extract_flag_emojis(df_spanish)
        print(f"Found {len(df_flags)} Spanish comments with flag emojis")
        df_flags[columns_to_save].to_csv(os.path.join(output_dir, f"{prefix}flag_emojis.csv"), index=False)
        
        # Combine all extracted Spanish comments
        df_all_spanish_cultural = pd.concat([
            df_cultural_keywords,
            df_cultural_identity, 
            df_religious_identity,
            df_locations,
            df_flags
        ]).drop_duplicates().reset_index(drop=True)
        
        print(f"\nTotal unique Spanish cultural comments: {len(df_all_spanish_cultural)}")
        df_all_spanish_cultural[columns_to_save].to_csv(os.path.join(output_dir, f"{prefix}all_cultural_comments.csv"), index=False)
        
        # Temporal analysis of Spanish cultural comments
        print("\n6. Analyzing temporal distribution...")
        temporal_dist = analyze_spanish_temporal_distribution(df_all_spanish_cultural)
        if not temporal_dist.empty:
            print("Spanish Cultural Comments Temporal Distribution:")
            print(temporal_dist)
            temporal_dist.to_csv(os.path.join(output_dir, f"{prefix}temporal_distribution.csv"))
        
        print(f"\nSpanish cultural analysis complete for {file_path}")
        print(f"Results saved with prefix: {prefix}")

    print("\n=== Spanish Cultural Analysis Summary ===")
    print("✅ Processed Spanish comments without translation")
    print("✅ Preserved cultural and linguistic authenticity")
    print("✅ Ready for topic modeling and cosine similarity analysis")

if __name__ == '__main__':
    main() 