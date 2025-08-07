"""
Utility functions for Cultural Analysis Project
Common functions used across multiple analysis tasks
"""

import pandas as pd
import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from config import VALIDATION_CONFIG

# === LOGGING SETUP ===
def setup_logging(log_level=logging.INFO):
    """Setup consistent logging across all modules"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cultural_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# === DATA LOADING & VALIDATION ===
def load_and_validate_data(file_path: Path, required_columns: List[str]) -> pd.DataFrame:
    """
    Load CSV file and validate required columns exist
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        Validated DataFrame
    """
    try:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
            
        # Read header first to check columns
        df_cols = pd.read_csv(file_path, nrows=0, low_memory=False).columns.tolist()
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df_cols]
        if missing_cols:
            logger.warning(f"Missing columns in {file_path}: {missing_cols}")
        
        # Load with available columns
        available_cols = [col for col in required_columns if col in df_cols]
        if available_cols:
            df = pd.read_csv(file_path, usecols=available_cols, low_memory=False)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        
        # Basic validation
        df = validate_dataframe(df)
        
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic data quality validation
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # Remove rows with missing comments
    if 'original_comment' in df.columns:
        df = df.dropna(subset=['original_comment'])
        df = df[df['original_comment'].str.len() >= VALIDATION_CONFIG['data_quality']['min_comment_length']]
        df = df[df['original_comment'].str.len() <= VALIDATION_CONFIG['data_quality']['max_comment_length']]
    
    # Validate timestamps
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df = df.dropna(subset=['published_at'])
        
        # Filter valid years
        valid_years = VALIDATION_CONFIG['data_quality']['valid_years']
        df = df[df['published_at'].dt.year.isin(valid_years)]
    
    # Handle spam labels
    if 'spam_label' in df.columns:
        df['spam_label'] = df['spam_label'].fillna('unknown')
    
    # Handle emotion labels  
    if 'emotion_label' in df.columns:
        df['emotion_label'] = df['emotion_label'].fillna('unknown')
    
    final_rows = len(df)
    if final_rows < initial_rows:
        logger.info(f"Data validation: {initial_rows} → {final_rows} rows ({initial_rows-final_rows} removed)")
    
    return df

# === TEXT PROCESSING UTILITIES ===
def clean_text(text: str) -> str:
    """Clean text for analysis"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text

def extract_words(text: str) -> List[str]:
    """Extract clean words from text"""
    if not isinstance(text, str):
        return []
    
    # Split and clean words
    words = text.lower().split()
    words = [re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word) for word in words]
    return [word for word in words if word and len(word) > 1]

def calculate_text_metrics(text: str) -> Dict[str, float]:
    """Calculate basic text metrics"""
    words = extract_words(text)
    
    if not words:
        return {
            'word_count': 0,
            'avg_word_length': 0,
            'lexical_diversity': 0,
            'uppercase_proportion': 0,
            'repeated_chars_count': 0
        }
    
    return {
        'word_count': len(words),
        'avg_word_length': sum(len(word) for word in words) / len(words),
        'lexical_diversity': len(set(words)) / len(words),
        'uppercase_proportion': sum(1 for char in text if char.isupper()) / len(text) if text else 0,
        'repeated_chars_count': len(re.findall(r'(.)\1{2,}', text))
    }

# === FILE UTILITIES ===
def ensure_directory_exists(file_path: Path) -> None:
    """Ensure parent directory exists for file path"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, file_path: Path, **kwargs) -> bool:
    """Safely save DataFrame to CSV with error handling"""
    try:
        ensure_directory_exists(file_path)
        df.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Saved {len(df)} rows to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        return False

def get_file_info(file_path: Path) -> Dict[str, any]:
    """Get file information for logging/reporting"""
    if not file_path.exists():
        return {'exists': False}
    
    stat = file_path.stat()
    return {
        'exists': True,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime
    }

# === PATTERN MATCHING UTILITIES ===
def create_regex_pattern(keywords: List[str], case_sensitive: bool = False) -> re.Pattern:
    """Create compiled regex pattern from keyword list"""
    escaped_keywords = [re.escape(keyword) for keyword in keywords]
    pattern = "|".join(escaped_keywords)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)

def extract_matching_comments(df: pd.DataFrame, pattern: re.Pattern, 
                            column: str = 'original_comment') -> pd.DataFrame:
    """Extract comments matching regex pattern"""
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return pd.DataFrame()
    
    matches = df[df[column].str.contains(pattern, na=False)]
    logger.info(f"Pattern matched {len(matches)} comments from {len(df)} total")
    return matches

# === TEMPORAL ANALYSIS UTILITIES ===
def analyze_temporal_distribution(df: pd.DataFrame, 
                                timestamp_col: str = 'published_at',
                                freq: str = 'M') -> pd.Series:
    """
    Analyze temporal distribution of comments
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        freq: Frequency for grouping ('M'=monthly, 'Y'=yearly, 'D'=daily)
        
    Returns:
        Series with temporal distribution
    """
    if df.empty or timestamp_col not in df.columns:
        return pd.Series(dtype='int64')
    
    try:
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy[timestamp_col], errors='coerce')
        df_copy = df_copy.dropna(subset=['timestamp'])
        
        df_copy['period'] = df_copy['timestamp'].dt.to_period(freq)
        distribution = df_copy['period'].value_counts().sort_index()
        
        logger.info(f"Temporal analysis: {len(distribution)} periods with data")
        return distribution
        
    except Exception as e:
        logger.error(f"Error in temporal analysis: {e}")
        return pd.Series(dtype='int64')

# === PERFORMANCE MONITORING ===
def log_performance(func):
    """Decorator for logging function performance"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper

# === PROGRESS TRACKING ===
def create_progress_tracker(total: int, desc: str = "Processing"):
    """Create simple progress tracker for console output"""
    def update_progress(current: int):
        if total > 0:
            percent = (current / total) * 100
            if current % max(1, total // 20) == 0:  # Update every 5%
                logger.info(f"{desc}: {current}/{total} ({percent:.1f}%)")
    
    return update_progress 