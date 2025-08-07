import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# Load the data
file_path = "parquet/yearly_data/despa_2018_en_nospam_labelled_default.parquet"
df = pd.read_parquet(file_path)

# Check data shape and columns
print(f"Data shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Identify the text column - assuming it's called 'text' or 'comment'
# If the column name is different, this needs to be updated
text_column = 'text' if 'text' in df.columns else 'comment'

if text_column not in df.columns:
    print(f"Text column not found. Available columns: {df.columns.tolist()}")
    # Try to identify a text column
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.len().mean() > 10:
            text_column = col
            print(f"Using {col} as text column")
            break

# Remove rows with empty text
df = df.dropna(subset=[text_column])
print(f"Data shape after dropping NA: {df.shape}")

# Get the documents
documents = df[text_column].tolist()

# Print sample documents
print("\nSample documents:")
for i in range(min(5, len(documents))):
    print(f"{i+1}. {documents[i][:100]}...")

# Create embeddings model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create UMAP model
umap_model = UMAP(n_neighbors=15, 
                 n_components=5, 
                 min_dist=0.0, 
                 metric='cosine', 
                 random_state=42)

# Create HDBSCAN model
hdbscan_model = HDBSCAN(min_cluster_size=15,
                       min_samples=10,
                       metric='euclidean',
                       cluster_selection_method='eom',
                       prediction_data=True)

# Create CountVectorizer
vectorizer = CountVectorizer(stop_words="english")

# Create and fit BERTopic model
topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    calculate_probabilities=True,  # Add this line
    verbose=True
)

# Fit the model to the documents
topics, probs = topic_model.fit_transform(documents)

# --- Debugging Start ---
print(f"\n--- Debugging Info ---")
print(f"Shape of topics: {np.shape(topics)}")
print(f"Shape of probabilities: {np.shape(probs)}")
print(f"Number of unique topics found: {len(np.unique(topics))}")
print(f"Topic distribution:\n{pd.Series(topics).value_counts()}")
print(f"--- End Debugging Info ---\n")
# --- Debugging End ---

# Get the topic info
topic_info = topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info.head(10))

# Get the most frequent topics
print("\nMost frequent topics:")
for topic in topic_info.head(10)["Topic"]:
    if topic != -1:  # Skip outlier topic
        print(f"Topic {topic}: {topic_model.get_topic(topic)}")

# Visualize the topics
print("\nGenerating visualizations...")

try:
    # Topic word scores visualization
    fig1 = topic_model.visualize_topics()
    fig1.write_html("topic_word_scores.html")
    print("Created: topic_word_scores.html")
except Exception as e:
    print(f"Error generating topic_word_scores.html: {e}")

try:
    # Topic hierarchy visualization
    fig2 = topic_model.visualize_hierarchy()
    fig2.write_html("topic_hierarchy.html")
    print("Created: topic_hierarchy.html")
except Exception as e:
    print(f"Error generating topic_hierarchy.html: {e}")

try:
    # Topic similarity visualization
    fig3 = topic_model.visualize_heatmap()
    fig3.write_html("topic_similarity.html")
    print("Created: topic_similarity.html")
except Exception as e:
    print(f"Error generating topic_similarity.html: {e}")

try:
    # Topic distribution bar plot
    fig4 = topic_model.visualize_barchart(top_n_topics=10)
    fig4.write_html("topic_distribution.html")
    print("Created: topic_distribution.html")
except Exception as e:
    print(f"Error generating topic_distribution.html: {e}")

# Add topics to the dataframe
df["topic"] = topics
df["topic_probability"] = probs.max(axis=1)
df["topic_name"] = df["topic"].apply(lambda x: topic_model.get_topic_info().loc[topic_model.get_topic_info()["Topic"] == x, "Name"].values[0] if x in topic_model.get_topic_info()["Topic"].values else "Unknown")

# Save the dataframe with topics
df.to_parquet("comments_with_topics.parquet")
print("Saved enriched data to: comments_with_topics.parquet")

print("\nBERTopic analysis completed successfully!")