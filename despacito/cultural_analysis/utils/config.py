"""
Configuration file for Cultural Analysis Project
Centralizes constants, file paths, and settings used across all modules
"""

from pathlib import Path
import os

# === PROJECT STRUCTURE ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "final_extraction"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "despa_outputs").mkdir(exist_ok=True)
(OUTPUT_DIR / "seeyou_outputs").mkdir(exist_ok=True)

# === INPUT FILES ===
INPUT_FILES = {
    'despa': DATA_DIR / "despa_kJQP7kiw5Fk_comments_spam_emotion.csv",
    'seeyou': DATA_DIR / "seeyou_RgKAFK5djSk_comments_spam_emotion.csv"
}

# === TASK 1: GEOGRAPHIC ANALYSIS ===
TASK1_CONFIG = {
    'keywords': [
        "hi from", "love from", "greetings from", "watching from",
        "sending love from", "greetings to", "hello from"
    ],
    'flag_emoji_pattern': r"[🇦-🇿]{2}",
    'location_pattern': r"\b(India|Turkey|USA|UK|Canada|Australia|Germany|France|Brazil|Japan|China|Russia|Italy|Spain|Mexico|Argentina|Egypt|Nigeria|Kenya|Pakistan|Bangladesh|[A-Za-z]+[Cc]ity)\b",
    'output_files': {
        'keyword': "_task_1_keyword_extracted_comments.csv",
        'flag': "_task_1_flag_emoji_extracted_comments.csv", 
        'location': "_task_1_location_extracted_comments.csv",
        'all': "_task_1_all_extracted_comments.csv",
        'temporal': "_task_1_temporal_distribution.csv"
    }
}

# === TASK 2: CULTURAL IDENTITY ===
TASK2_CONFIG = {
    'categories': {
        'Ethnic Identity': ['latino', 'hispanic', 'asian', 'white', 'caucasian', 'black', 'african'],
        'Religious Identity': ['muslim', 'islam', 'christian', 'christ', 'buddhist', 'hindu'],
        'Collective Memory': ['my country', 'our country', 'in my culture', 'homeland', 'diaspora', 'immigrant']
    },
    'context_keywords': [
        'muslim', 'latino', 'in my country', 'where i\'m from', 'asian',
        'hispanic', 'christian', 'homeland', 'diaspora'
    ],
    'sample_size': 20,
    'output_files': {
        'filtered': "_task_2_context_filtered_comments.csv",
        'sample': "_task_2_sample_for_manual_analysis.csv",
        'labeled': "_task_2_labeled_context_comments.csv"
    }
}

# === TASK 3: LINGUISTIC ANALYSIS ===
TASK3_CONFIG = {
    'scoring_weights': {
        'basic': {
            'avg_word_length': 0.2,
            'lexical_diversity': 0.4,
            'uppercase_proportion': -0.2,
            'repeated_chars': -0.2
        },
        'enhanced': {
            'readability': 0.4,
            'grammar_errors': -0.3,
            'lexical_diversity': 0.15,
            'avg_word_length': 0.05,
            'uppercase_proportion': -0.05,
            'repeated_chars': -0.05
        }
    },
    'multiprocessing': {
        'max_processes': 6,
        'chunk_size': 100
    },
    'output_files': {
        'basic': "_task_3_basic_linguistic_features.csv",
        'enhanced': "_task_3_enhanced_linguistic_analysis.csv"
    }
}

# === VISUALIZATION SETTINGS ===
VIZ_CONFIG = {
    'colors': {
        'ethnic_identity': '#FF6B6B',
        'religious_identity': '#4ECDC4', 
        'collective_memory': '#45B7D1',
        'despa': '#FF6B6B',
        'seeyou': '#4ECDC4'
    },
    'chart_defaults': {
        'height': 600,
        'font_size': 12,
        'background_color': 'rgba(0,0,0,0)'
    }
}

# === FILE NAMING CONVENTIONS ===
def get_file_prefix(file_path):
    """Generate appropriate prefix from file path"""
    base_name = os.path.basename(str(file_path))
    if "despa_" in base_name:
        return "despa"
    elif "seeyou_" in base_name:
        return "seeyou"
    return "unknown"

def get_output_path(prefix, task, file_type):
    """Generate standardized output file paths"""
    if task == 1:
        return OUTPUT_DIR / f"{prefix}_outputs" / f"{prefix}{TASK1_CONFIG['output_files'][file_type]}"
    elif task == 2:
        return OUTPUT_DIR / f"{prefix}_outputs" / f"{prefix}{TASK2_CONFIG['output_files'][file_type]}"
    elif task == 3:
        return OUTPUT_DIR / f"{prefix}_outputs" / f"{prefix}{TASK3_CONFIG['output_files'][file_type]}"
    else:
        raise ValueError(f"Unknown task: {task}")

# === VALIDATION SETTINGS ===
VALIDATION_CONFIG = {
    'required_columns': {
        'base': ['original_comment', 'published_at'],
        'spam': ['spam_label'],
        'emotion': ['emotion_label'],
        'language': ['detected_language']
    },
    'data_quality': {
        'min_comment_length': 3,
        'max_comment_length': 10000,
        'valid_years': range(2005, 2026)
    }
} 