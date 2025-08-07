import pandas as pd
import json
import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# File paths
file_path = 'seeyou_comments_noemoji.csv'  # Input CSV file
output_path = 'seeyou_comments_noemoji_naive.csv'  # Output CSV file
progress_path = 'sentiment_progress2.json'  # JSON file to track progress

print(f"Starting sentiment analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input file: {file_path}")
print(f"Output file: {output_path}")

# Load the dataset without headers
print("Loading dataset...")
df = pd.read_csv(file_path, header=None)
print(f"Loaded {len(df)} comments")

# The comment text is in the second column (index 1)
df[1] = df[1].astype(str)

# Load progress or initialize
if os.path.exists(progress_path):
    with open(progress_path, 'r') as progress_file:
        progress_data = json.load(progress_file)
        start_index = progress_data.get('last_processed_index', 0)
    print(f"Resuming from index {start_index}")
else:
    start_index = 0
    print("Starting from beginning")

# Initialize counters
total_processed = 0
positive_comments = 0
negative_comments = 0
error_comments = 0

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = ''.join(c for c in text if c.isalpha() or c.isspace())
    return text

# Training data (you can modify these examples based on your needs)
training_data = [
    ("This is amazing! I love it!", "positive"),
    ("Great video, very informative", "positive"),
    ("Best content I've seen in a while", "positive"),
    ("This is terrible, I hate it", "negative"),
    ("Waste of time, very disappointing", "negative"),
    ("Not worth watching at all", "negative"),
    ("The quality is poor and unprofessional", "negative"),
    ("Excellent work, keep it up!", "positive"),
    ("I learned so much from this", "positive"),
    ("This is exactly what I needed", "positive"),
    ("Complete garbage, don't bother", "negative"),
    ("Could have been better", "negative"),
    ("Not what I expected, very bad", "negative"),
    ("Absolutely fantastic!", "positive"),
    ("Very helpful and well explained", "positive")
]

# Prepare training data
X_train = [preprocess_text(text) for text, _ in training_data]
y_train = [label for _, label in training_data]

# Initialize and train the vectorizer and classifier
print("Training the sentiment analyzer...")
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Function to determine sentiment
def analyze_sentiment(comment):
    try:
        # Preprocess the comment
        processed_comment = preprocess_text(comment)
        
        # Vectorize the comment
        comment_vector = vectorizer.transform([processed_comment])
        
        # Predict sentiment
        prediction = classifier.predict(comment_vector)[0]
        probability = classifier.predict_proba(comment_vector)[0]
        
        return prediction, probability
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return None, None

# Open the output file in append mode
with open(output_path, mode='a', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file)
    
    # Write the header row if this is a new file
    if start_index == 0 and not os.path.exists(output_path):
        writer.writerow(['comment', 'sentiment', 'positive_probability', 'negative_probability'])
        print("Created new output file with headers")

    # Process rows starting from the last saved index
    total_rows = len(df) - start_index
    print(f"Processing {total_rows} comments...")
    
    for index, row in df.iloc[start_index:].iterrows():
        total_processed += 1
        comment = row[1]  # Get comment from second column
        
        if total_processed % 1000 == 0:
            print(f"Processed {total_processed}/{total_rows} comments ({total_processed/total_rows*100:.1f}%)")
            print(f"Positive: {positive_comments}, Negative: {negative_comments}, Errors: {error_comments}")
        
        try:
            sentiment, probabilities = analyze_sentiment(comment)
            if sentiment is not None:
                writer.writerow([
                    comment,
                    sentiment,
                    probabilities[1],  # Positive probability
                    probabilities[0]   # Negative probability
                ])
                if sentiment == 'positive':
                    positive_comments += 1
                else:
                    negative_comments += 1
            else:
                error_comments += 1
        except Exception as e:
            print(f"Error processing comment at index {index}: {str(e)}")
            error_comments += 1
        
        # Update progress
        if total_processed % 100 == 0:
            with open(progress_path, 'w') as progress_file:
                json.dump({'last_processed_index': index}, progress_file)

print(f"\nProcessing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total comments processed: {total_processed}")
print(f"Positive comments: {positive_comments}")
print(f"Negative comments: {negative_comments}")
print(f"Comments with errors: {error_comments}")
print(f"Sentiment analysis results saved to {output_path}") 
