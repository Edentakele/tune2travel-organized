# Tune2Travel: YouTube Comment Cultural Analytics

YouTube comment analysis system processing "Despacito" and "See You Again" videos. Extracts geographic origins, cultural markers, emotions, and linguistic patterns.

## System Architecture

**Core Modules:**
- `cultural_analysis/`: Geographic/temporal/linguistic extraction
- `cultural_analysis_spanish/`: Spanish-only processing (no translation)
- `emotion_analysis/`: 7-class emotion classification + topic modeling
- `you-tune/`: CLI analysis tool
- `deprecated/`: Obsolete VPN/download scripts

**Data Pipeline:**
1. Comment extraction → spam filtering → emotion labeling → cultural analysis → visualization
2. Processes: `despa_kJQP7kiw5Fk_comments_spam_emotion.csv`, `seeyou_RgKAFK5djSk_comments_spam_emotion.csv`

## Execution

```bash
# Cultural analysis
cd cultural_analysis && python task_1_regex_keyword_extraction.py
python task_2_keyword_context_filtering.py
python task_3_linguistic_analysis_basic.py

# Spanish processing
cd cultural_analysis_spanish && python task_1_spanish_keyword_extraction.py
python spanish_bertopic_analysis.py

# Emotion analysis
cd emotion_analysis && python emotion_analyzer.py

# CLI tool
cd you-tune && python you-tune.py --input comments.csv --output analyzed.csv
```

## Capabilities

**Geographic Extraction:**
- Keywords: "love from", "hi from", "watching from"
- Flag emoji detection (🇮🇳, 🇹🇷, etc.)
- Location name matching (basic regex, limited accuracy)

**Cultural Analysis:**
- Ethnic markers: latino, hispanic, asian, black, white, caucasian
- Religious markers: muslim, christian, islam, christ
- Collective memory: "my country", "our country", "homeland", "diaspora"

**Emotion Classification:**
- 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise
- Model: j-hartmann/emotion-english-distilroberta-base
- Synthetic data generation for underrepresented classes

**Linguistic Analysis:**
- Basic: word length, lexical diversity, uppercase ratio, repeated characters
- Enhanced: grammar errors (LanguageTool), readability (Flesch)

**Spanish Processing:**
- No translation preserves cultural authenticity
- Multilingual BERT: paraphrase-multilingual-MiniLM-L12-v2
- Spanish stopwords, cultural keywords

## Limitations

**Data Quality:**
- Location extraction uses basic regex, misses many locations
- Manual labeling required for emotion ground truth
- Spam detection basic logistic regression

**Geographic Accuracy:**
- Country extraction incomplete, requires comprehensive location database
- Flag emoji detection limited to regional indicator symbols
- City detection minimal

**Emotion Model:**
- English-only for main pipeline
- Requires Java JDK 17+ for enhanced linguistic analysis
- Synthetic data may introduce training bias

**Performance:**
- Grammar checking computationally expensive
- GPU recommended for large datasets
- Memory constraints with full processing pipeline

## Dependencies

```bash
pip install pandas numpy torch transformers matplotlib seaborn plotly
pip install language-tool-python textstat bertopic emoji pycountry
```

**System Requirements:**
- Python 3.6+
- Java JDK 17+ (for enhanced linguistic analysis)
- CUDA-capable GPU (recommended)

## Output Files

**Cultural Analysis:**
- Geographic extraction CSVs per video
- Temporal distribution by year-month
- Interactive HTML visualizations

**Emotion Analysis:**
- Classification results with confidence scores
- Confusion matrices
- Synthetic training data

**Spanish Processing:**
- Topic modeling HTML files
- Cultural marker extraction
- Similarity matrices

## Technical Debt

- Hardcoded file paths throughout scripts
- Inconsistent error handling
- Limited test coverage
- Mixed Spanish/English processing logic
- Dependency management scattered 