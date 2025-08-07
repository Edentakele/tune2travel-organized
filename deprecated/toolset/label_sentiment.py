import pandas as pd
from transformers import pipeline
import time
from tqdm import tqdm
import os
import fasttext
import re
import numpy as np

def load_comments(file_path):
    """Load comments from CSV file"""
    print(f"Loading comments from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def initialize_sentiment_analyzer():
    """Initialize the sentiment analysis pipeline"""
    print("Initializing sentiment analyzer...")
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def initialize_language_detector():
    """Initialize the language detection model"""
    print("Initializing language detector...")
    return fasttext.load_model('lid.176.bin')

def clean_comment(comment):
    """Clean comment text for better language detection"""
    # Handle NaN and float values
    if pd.isna(comment) or isinstance(comment, float):
        return ""
    
    # Convert to string and lowercase
    comment = str(comment).lower()
    # Remove URLs
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
    # Remove special characters but keep spaces
    comment = re.sub(r'[^\w\s]', ' ', comment)
    # Remove extra whitespace
    comment = ' '.join(comment.split())
    return comment

def is_english(comment, lang_detector, min_confidence=0.7):
    """Check if a comment is in English with confidence threshold"""
    # Clean the comment first
    cleaned_comment = clean_comment(comment)
    if not cleaned_comment or len(cleaned_comment.strip()) < 3:
        return False
    
    try:
        # Get language prediction
        predictions = lang_detector.predict(cleaned_comment, k=1)
        language = predictions[0][0].replace('__label__', '')
        # Convert confidence to float using np.asarray
        confidence = float(np.asarray(predictions[1][0]))
        
        return language == 'en' and confidence >= min_confidence
    except Exception as e:
        print(f"Error in language detection: {str(e)}")
        return False

def filter_english_comments(df, output_file="youtube_comments_english.csv"):
    """Filter out non-English comments and save in real-time"""
    print("Filtering English comments...")
    
    # Initialize language detector
    lang_detector = initialize_language_detector()
    
    # Create output file with header
    df.iloc[0:0].to_csv(output_file, index=False)
    
    # Process in batches
    batch_size = 1000
    total_comments = len(df)
    english_count = 0
    
    for i in tqdm(range(0, total_comments, batch_size)):
        batch_end = min(i + batch_size, total_comments)
        batch_df = df.iloc[i:batch_end].copy()
        
        # Add language column
        batch_df['language'] = 'non-en'
        
        # Process each comment in the batch
        for j in range(len(batch_df)):
            comment = batch_df.iloc[j]['comment']
            if is_english(comment, lang_detector):
                batch_df.iloc[j, batch_df.columns.get_loc('language')] = 'en'
                english_count += 1
        
        # Save only English comments from this batch
        english_batch = batch_df[batch_df['language'] == 'en']
        if not english_batch.empty:
            english_batch.to_csv(output_file, mode='a', header=False, index=False)
    
    print(f"\nLanguage Statistics:")
    print(f"Total comments: {total_comments}")
    print(f"English comments: {english_count}")
    print(f"Non-English comments: {total_comments - english_count}")
    
    # Return only English comments for further processing
    return df[df.index.isin(range(total_comments)) & (df['language'] == 'en')]

def truncate_comment(comment, max_length=500):
    """Truncate comment to a reasonable length"""
    if pd.isna(comment) or isinstance(comment, float):
        return ""
    comment = str(comment)
    if len(comment) > max_length:
        return comment[:max_length] + "..."
    return comment

def analyze_sentiment(comments, analyzer, batch_size=100):
    """Analyze sentiment of comments in batches"""
    print("Analyzing sentiment...")
    sentiments = []
    
    # Process comments in batches
    for i in tqdm(range(0, len(comments), batch_size)):
        batch = comments[i:i + batch_size]
        # Ensure each comment is a string and truncate if too long
        batch = [truncate_comment(comment) for comment in batch]
        try:
            results = analyzer(batch, truncation=True, max_length=512)
            sentiments.extend(results)
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            # Add default sentiment for failed batch
            sentiments.extend([{'label': 'NEUTRAL', 'score': 0.5} for _ in batch])
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
    
    return sentiments

def save_results(df, sentiments, output_file):
    """Save results to CSV file"""
    print("Saving results...")
    
    # Add sentiment columns to dataframe
    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    # File paths
    input_file = "youtube_comments_combined.csv"
    english_file = "youtube_comments_english.csv"
    output_file = "youtube_comments_with_sentiment.csv"
    
    # Load comments
    df = load_comments(input_file)
    
    # Filter English comments and save in real-time
    df_english = filter_english_comments(df, english_file)
    
    # Initialize sentiment analyzer
    sentiment_analyzer = initialize_sentiment_analyzer()
    
    # Get comments as list and ensure they are strings
    comments = df_english['comment'].fillna('').astype(str).tolist()
    
    # Analyze sentiment
    sentiments = analyze_sentiment(comments, sentiment_analyzer)
    
    # Save results
    save_results(df_english, sentiments, output_file)
    
    # Print summary
    sentiment_counts = df_english['sentiment_label'].value_counts()
    print("\nSentiment Distribution:")
    print(sentiment_counts)

if __name__ == "__main__":
    main() 