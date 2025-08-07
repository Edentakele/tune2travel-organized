# Spanish Comment Analysis

Spanish-only processing for Despacito comments. No translation to preserve cultural authenticity.

## Rationale

Translation destroys cultural context. Spanish expressions like "saludos desde", "soy latino", "mi país" lose meaning when translated. Multilingual models handle Spanish adequately without translation overhead.

## Implementation

### Spanish Cultural Extraction

**Script:** `task_1_spanish_keyword_extraction.py`

**Spanish Keywords:**
- Geographic: "saludos desde", "amor desde", "soy de"
- Identity: "mexicano", "argentino", "colombiano", "soy latino"
- Religious: "católico", "cristiano", "musulmán"
- Cultural: "mi país", "nuestra cultura", "patria"

**Process:** Regex matching on Spanish comment subset

### Topic Modeling

**Script:** `spanish_bertopic_analysis.py`

**Technical Stack:**
- Model: paraphrase-multilingual-MiniLM-L12-v2
- Spanish stopwords: 70+ common words
- UMAP: n_neighbors=15, metric=cosine
- HDBSCAN: min_cluster_size=10, min_samples=5
- CountVectorizer: ngram_range=(1,2)

## Outputs

**Cultural Extraction:**
- `despa_spanish_cultural_keywords.csv`
- `despa_spanish_cultural_identity.csv`
- `despa_spanish_religious_identity.csv`
- `despa_spanish_locations.csv`
- `despa_spanish_flag_emojis.csv`
- `despa_spanish_all_cultural_comments.csv`
- `despa_spanish_temporal_distribution.csv`

**Topic Analysis:**
- `spanish_despa_topic_word_scores.html`
- `spanish_despa_topic_hierarchy.html`
- `spanish_despa_topic_similarity.html`
- `spanish_despa_topic_distribution.html`
- `spanish_despa_comments_with_topics.csv`
- `spanish_despa_topic_similarities.json`

## Expected Results

**Topic Categories:**
- Latin identity expressions
- Geographic homeland references
- Reggaeton/music appreciation
- Diaspora community sentiment

## Quality Indicators

**Good Results:**
- Coherent Spanish phrase clustering
- Logical geographic patterns
- Preserved cultural expressions

**Warning Signs:**
- Language mixing in topics
- Incoherent Spanish combinations
- Lost cultural context markers

## Limitations

**Cultural Coverage:**
- Limited to predefined keyword sets
- No semantic understanding of cultural nuances
- Regional Spanish variations not distinguished

**Technical Constraints:**
- Requires substantial Spanish comment volume
- Topic quality depends on clustering parameters
- No validation against cultural expertise

## Execution

```bash
cd cultural_analysis_spanish
python task_1_spanish_keyword_extraction.py
python spanish_bertopic_analysis.py
```

## Integration

Compatible with existing pipeline:
- Uses language detection results from main system
- Maintains spam/emotion label compatibility
- Follows established file naming conventions 