# Cultural Analysis Module

Geographic, temporal, and linguistic signal extraction from YouTube comments. Processes two datasets: Despacito and See You Again.

## Input Requirements

Fixed input files:
- `../final_extraction/despa_kJQP7kiw5Fk_comments_spam_emotion.csv`
- `../final_extraction/seeyou_RgKAFK5djSk_comments_spam_emotion.csv`

Required columns: `original_comment`, `published_at`, `spam_label`, `emotion_label`

## Task 1: Geographic Extraction

**Script:** `task_1_regex_keyword_extraction.py`

**Geographic Markers:**
- Keywords: "hi from", "love from", "greetings from", "watching from"
- Flag emojis: Unicode Regional Indicator Symbols (🇦-🇿 pairs)
- Location names: Hardcoded list of countries/major cities

**Limitations:**
- Location regex extremely basic, misses most locations
- No geocoding or NER integration
- Flag emoji detection limited to standard Unicode ranges
- No validation of location accuracy

**Outputs:**
- `[prefix]task_1_keyword_extracted_comments.csv`
- `[prefix]task_1_flag_emoji_extracted_comments.csv`
- `[prefix]task_1_location_extracted_comments.csv`
- `[prefix]task_1_all_extracted_comments.csv`
- `[prefix]task_1_temporal_distribution.csv`

## Task 2: Cultural Context Filtering

**Script:** `task_2_keyword_context_filtering.py`

**Cultural Keywords:**
- Ethnic: "Latino", "Hispanic", "Asian", "White", "Caucasian", "Black"
- Religious: "Muslim", "Islam", "Christian", "Christ"
- Collective: "my country", "our country", "in my culture", "homeland", "diaspora", "immigrant"

**Process:**
- Regex matching (case-insensitive)
- Random sampling of 20 comments per video for manual review
- No context validation or semantic analysis

**Outputs:**
- `[prefix]task_2_context_filtered_comments.csv`
- `[prefix]task_2_sample_for_manual_analysis.csv`

## Task 3: Linguistic Analysis

### Basic Version (Fast, Inaccurate)

**Script:** `task_3_linguistic_analysis_basic.py`

**Metrics:**
- Average word length
- Lexical diversity (unique/total words)
- Uppercase character proportion
- Repeated character sequences count

**Scoring Formula:**
```python
score = (avg_word_length * 0.2) + (lexical_diversity * 0.4) - 
        (uppercase_proportion * 0.2) - (log(repeated_chars + 1) * 0.2)
```

**Reality Check:**
- Scoring weights arbitrary, not validated
- No correlation with actual language quality
- Heuristic approach without linguistic foundation

### Enhanced Version (Slow, Better)

**Dependencies:** Java JDK 17+, language-tool-python, textstat

**Additional Metrics:**
- Grammar error count (LanguageTool)
- Flesch Reading Ease score
- Multiprocessing for performance

**Limitations:**
- LanguageTool accuracy varies by language
- Computationally expensive for large datasets
- Grammar checking quality depends on text length
- Enhanced score weights still arbitrary

## System Constraints

**Performance:**
- No optimization for large datasets
- Sequential processing only
- Memory usage scales linearly with input size

**Accuracy:**
- Geographic extraction misses majority of locations
- Cultural keyword matching crude and incomplete
- Linguistic scoring experimental and unvalidated

**Maintenance:**
- Hardcoded file paths
- No configuration management
- Limited error handling
- No logging system

## Execution

```bash
# Basic pipeline
python task_1_regex_keyword_extraction.py
python task_2_keyword_context_filtering.py
python task_3_linguistic_analysis_basic.py

# Enhanced linguistic analysis (requires Java)
python task_3_linguistic_analysis.py
```

## Expected Accuracy

**Geographic Extraction:** 15-30% recall for actual locations
**Cultural Keywords:** 60-80% precision, unknown recall
**Linguistic Analysis:** Basic version correlation with quality: poor; Enhanced version: moderate 