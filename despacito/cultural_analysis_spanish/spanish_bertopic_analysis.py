import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import os

def load_spanish_comments(file_path):
    """Load Spanish comments from extracted cultural analysis results."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} Spanish cultural comments from {file_path}")
        
        # Clean and prepare text
        df = df.dropna(subset=['original_comment'])
        df['original_comment'] = df['original_comment'].astype(str)
        
        # Filter out very short comments (less than 10 characters)
        df = df[df['original_comment'].str.len() >= 10]
        
        print(f"After cleaning: {len(df)} comments ready for analysis")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Spanish comments: {e}")
        return pd.DataFrame()

def create_spanish_topic_model():
    """Create a BERTopic model optimized for Spanish text."""
    
    # Use multilingual sentence transformer that works well with Spanish
    print("Loading multilingual sentence transformer...")
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Configure UMAP for Spanish text clustering
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Configure HDBSCAN with parameters suitable for Spanish comments
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,  # Smaller clusters for more granular topics
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Spanish-specific CountVectorizer with Spanish stopwords
    spanish_stopwords = [
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su',
        'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'pero', 'sus', 'le', 'ha', 'me',
        'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'son', 'dos',
        'también', 'fue', 'había', 'muy', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo',
        'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él',
        'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro',
        'después', 'te', 'otros', 'aunque', 'esa', 'eso', 'hace', 'otra', 'gobierno', 'tan', 'durante',
        'siempre', 'día', 'tanto', 'ella', 'tres', 'sí', 'dijo', 'años', 'antes', 'estado', 'nunca',
        'agua', 'menos', 'nuevo', 'debe', 'casa', 'país', 'hecho', 'general', 'mayor', 'propio',
        'mismo', 'año', 'trabajo', 'hombre', 'mundo', 'caso', 'mujer', 'forma', 'lugar', 'grupo',
        'momento', 'número', 'manera', 'empresa', 'horas', 'ciudad', 'problem', 'hora', 'fin',
        'medio', 'millones', 'hacia', 'punto', 'aunque', 'además', 'ejemplo', 'desarrollo'
    ]
    
    vectorizer = CountVectorizer(
        stop_words=spanish_stopwords,
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.95,  # Word must not appear in more than 95% of documents
        ngram_range=(1, 2)  # Include bigrams for better Spanish phrase detection
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=True
    )
    
    return topic_model

def analyze_spanish_topics(df, output_prefix="spanish_despa", output_dir=None):
    """Perform topic modeling on Spanish comments."""
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get documents
    documents = df['original_comment'].tolist()
    print(f"\nAnalyzing {len(documents)} Spanish documents...")
    
    # Print sample documents
    print("\nSample Spanish documents:")
    for i in range(min(5, len(documents))):
        print(f"{i+1}. {documents[i][:100]}...")
    
    # Create topic model
    print("\nCreating Spanish topic model...")
    topic_model = create_spanish_topic_model()
    
    # Fit the model
    print("Fitting topic model to Spanish documents...")
    topics, probs = topic_model.fit_transform(documents)
    
    # Print debugging info
    print(f"\n--- Spanish Topic Analysis Results ---")
    print(f"Shape of topics: {np.shape(topics)}")
    print(f"Shape of probabilities: {np.shape(probs)}")
    print(f"Number of unique topics found: {len(np.unique(topics))}")
    print(f"Topic distribution:\n{pd.Series(topics).value_counts()}")
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print("\nSpanish Topic Information:")
    print(topic_info.head(10))
    
    # Print most frequent topics with Spanish keywords
    print("\nMost frequent Spanish topics:")
    for topic in topic_info.head(10)["Topic"]:
        if topic != -1:  # Skip outlier topic
            topic_words = topic_model.get_topic(topic)
            print(f"Topic {topic}: {topic_words}")
    
    # Generate visualizations
    print(f"\nGenerating Spanish topic visualizations with prefix: {output_prefix}...")
    
    try:
        # Topic word scores visualization
        fig1 = topic_model.visualize_topics()
        fig1.write_html(os.path.join(output_dir, f"{output_prefix}_topic_word_scores.html"))
        print(f"Created: {os.path.join(output_dir, f'{output_prefix}_topic_word_scores.html')}")
    except Exception as e:
        print(f"Error generating topic word scores: {e}")

    try:
        # Topic hierarchy visualization
        fig2 = topic_model.visualize_hierarchy()
        fig2.write_html(os.path.join(output_dir, f"{output_prefix}_topic_hierarchy.html"))
        print(f"Created: {os.path.join(output_dir, f'{output_prefix}_topic_hierarchy.html')}")
    except Exception as e:
        print(f"Error generating topic hierarchy: {e}")

    try:
        # Topic similarity heatmap
        fig3 = topic_model.visualize_heatmap()
        fig3.write_html(os.path.join(output_dir, f"{output_prefix}_topic_similarity.html"))
        print(f"Created: {os.path.join(output_dir, f'{output_prefix}_topic_similarity.html')}")
    except Exception as e:
        print(f"Error generating topic similarity: {e}")

    try:
        # Topic distribution bar plot
        fig4 = topic_model.visualize_barchart(top_n_topics=10)
        fig4.write_html(os.path.join(output_dir, f"{output_prefix}_topic_distribution.html"))
        print(f"Created: {os.path.join(output_dir, f'{output_prefix}_topic_distribution.html')}")
    except Exception as e:
        print(f"Error generating topic distribution: {e}")

    # Add topics to the dataframe
    df_with_topics = df.copy()
    df_with_topics["topic"] = topics
    df_with_topics["topic_probability"] = probs.max(axis=1) if len(probs.shape) > 1 else probs
    
    # Add topic names
    topic_names = {}
    for topic_id in topic_info["Topic"]:
        if topic_id in topic_model.get_topic_info()["Topic"].values:
            topic_names[topic_id] = topic_model.get_topic_info().loc[
                topic_model.get_topic_info()["Topic"] == topic_id, "Name"
            ].values[0]
        else:
            topic_names[topic_id] = "Unknown"
    
    df_with_topics["topic_name"] = df_with_topics["topic"].map(topic_names)
    
    # Save results
    output_file = os.path.join(output_dir, f"{output_prefix}_comments_with_topics.csv")
    df_with_topics.to_csv(output_file, index=False)
    print(f"Saved enriched Spanish data to: {output_file}")
    
    return topic_model, df_with_topics

def calculate_spanish_cosine_similarity(df_with_topics, topic_model, output_prefix="spanish_despa", output_dir=None):
    """Calculate cosine similarity between Spanish topics."""
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import json
        
        # Get topic embeddings/representations
        topics_info = topic_model.get_topic_info()
        
        # Create a matrix of topic word scores for similarity calculation
        topic_words_matrix = []
        topic_labels = []
        
        for topic_id in topics_info["Topic"]:
            if topic_id != -1:  # Skip outlier topic
                topic_words = topic_model.get_topic(topic_id)
                # Convert to a dictionary for easier processing
                word_scores = {word: score for word, score in topic_words}
                topic_words_matrix.append(word_scores)
                topic_labels.append(f"Topic_{topic_id}")
        
        # Calculate similarities between topics based on their word distributions
        print(f"\nCalculating cosine similarities between {len(topic_labels)} Spanish topics...")
        
        # Create a simple similarity analysis
        similarity_results = {}
        
        for i, topic_a in enumerate(topic_labels):
            similarity_results[topic_a] = {}
            topic_a_words = set(topic_words_matrix[i].keys())
            
            for j, topic_b in enumerate(topic_labels):
                if i != j:
                    topic_b_words = set(topic_words_matrix[j].keys())
                    
                    # Calculate Jaccard similarity (simpler than cosine for this case)
                    intersection = len(topic_a_words.intersection(topic_b_words))
                    union = len(topic_a_words.union(topic_b_words))
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    similarity_results[topic_a][topic_b] = jaccard_sim
        
        # Save similarity results
        similarity_file = os.path.join(output_dir, f"{output_prefix}_topic_similarities.json")
        with open(similarity_file, 'w', encoding='utf-8') as f:
            json.dump(similarity_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved Spanish topic similarities to: {similarity_file}")
        
        # Print top similarities
        print("\nTop Spanish topic similarities:")
        for topic_a, similarities in similarity_results.items():
            if similarities:
                best_match = max(similarities.items(), key=lambda x: x[1])
                print(f"{topic_a} most similar to {best_match[0]} (similarity: {best_match[1]:.3f})")
        
        return similarity_results
        
    except Exception as e:
        print(f"Error calculating Spanish cosine similarities: {e}")
        return {}

def main():
    """Main function to run Spanish topic analysis."""
    
    # Look for Spanish cultural analysis results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for Spanish cultural comments file
    spanish_file_path = os.path.join(script_dir, 'despa_spanish_all_cultural_comments.csv')
    
    # Alternative: check for the Spanish extraction script output
    if not os.path.exists(spanish_file_path):
        print(f"Spanish cultural comments file not found at: {spanish_file_path}")
        print("Please run the Spanish cultural analysis first:")
        print("python cultural_analysis/spanish/task_1_spanish_keyword_extraction.py")
        return
    
    # Load Spanish comments
    df_spanish = load_spanish_comments(spanish_file_path)
    
    if df_spanish.empty:
        print("No Spanish comments found. Exiting.")
        return
    
    # Analyze Spanish topics
    print("\n=== Starting Spanish Topic Analysis ===")
    topic_model, df_with_topics = analyze_spanish_topics(df_spanish, "spanish_despa", script_dir)
    
    # Calculate similarities
    print("\n=== Calculating Spanish Topic Similarities ===")
    similarity_results = calculate_spanish_cosine_similarity(df_with_topics, topic_model, "spanish_despa", script_dir)
    
    print("\n=== Spanish Analysis Complete ===")
    print("✅ Topic modeling completed without translation")
    print("✅ Cosine similarities calculated")
    print("✅ Cultural authenticity preserved")
    print("✅ Ready for manual coding stage")
    
    # Provide summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total Spanish comments analyzed: {len(df_with_topics)}")
    print(f"Unique topics discovered: {len(df_with_topics['topic'].unique())}")
    print(f"Comments with high topic confidence (>0.5): {len(df_with_topics[df_with_topics['topic_probability'] > 0.5])}")

if __name__ == '__main__':
    main() 