import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter
import re

def clean_text(text):
    """Clean and normalize text for word frequency analysis"""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_file(file_path):
    """Analyze a labeled CSV file and generate visualizations"""
    print(f"Analyzing {file_path}...")
    
    # Read the labeled data
    df = pd.read_csv(file_path)
    
    # Get the year from the filename
    year = os.path.basename(file_path).split('_')[-1].split('.')[0]
    
    # Create a directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Distribution of tourism vs non-tourism comments
    tourism_count = df['tourism_related'].sum()
    non_tourism_count = len(df) - tourism_count
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Tourism Related', 'Not Tourism Related'], [tourism_count, non_tourism_count])
    plt.title(f'Distribution of Comments ({year})')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.savefig(f'visualizations/distribution_{year}.png')
    plt.close()
    
    # 2. Word frequency analysis for tourism-related comments
    if tourism_count > 0:
        tourism_comments = df[df['tourism_related'] == 1]['comment'].apply(clean_text)
        
        # Extract words and count frequency
        all_words = ' '.join(tourism_comments).split()
        word_counts = Counter(all_words)
        
        # Remove common stop words
        stop_words = ['the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'for', 'with', 'this', 'that', 
                      'of', 'on', 'at', 'by', 'from', 'as', 'was', 'were', 'be', 'been', 'being', 
                      'am', 'are', 'i', 'you', 'he', 'she', 'they', 'we', 'me', 'my', 'your', 'his', 
                      'her', 'their', 'our', 'its', 'de', 'la', 'el', 'en', 'y', 'que', 'los', 'las',
                      'es', 'un', 'una', 'del', 'se', 'por', 'con', 'no', 'si', 'al', 'lo', 'como',
                      'mas', 'pero', 'sus', 'le', 'ya', 'o', 'muy', 'sin', 'sobre', 'mi', 'tu', 'me',
                      'hasta', 'porque', 'cuando', 'quien', 'donde', 'solo', 'yo', 'te', 'esta',
                      'este', 'estos', 'estas', 'aquel']
        
        for word in list(word_counts.keys()):
            if word in stop_words or len(word) < 3:
                del word_counts[word]
        
        # Get the top 15 words
        top_words = word_counts.most_common(15)
        
        plt.figure(figsize=(12, 6))
        plt.bar([word for word, count in top_words], [count for word, count in top_words])
        plt.title(f'Most Common Words in Tourism-Related Comments ({year})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'visualizations/word_frequency_{year}.png')
        plt.close()
    
    # Print summary statistics
    print(f"Total comments: {len(df)}")
    print(f"Tourism-related comments: {tourism_count} ({tourism_count/len(df)*100:.2f}%)")
    print(f"Non-tourism-related comments: {non_tourism_count} ({non_tourism_count/len(df)*100:.2f}%)")
    
    return {
        'year': year,
        'total': len(df),
        'tourism': tourism_count,
        'non_tourism': non_tourism_count
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize results of YouTube comment labeling')
    parser.add_argument('--file', '-f', help='Labeled CSV file to analyze')
    parser.add_argument('--dir', '-d', help='Directory of labeled CSV files to analyze')
    
    args = parser.parse_args()
    
    results = []
    
    if args.file:
        results.append(analyze_file(args.file))
    elif args.dir:
        for filename in os.listdir(args.dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(args.dir, filename)
                results.append(analyze_file(file_path))
    else:
        # Check the current directory for labeled_*.csv files
        for filename in os.listdir('.'):
            if filename.startswith('labeled_') and filename.endswith('.csv'):
                results.append(analyze_file(filename))
    
    # Create a year-by-year comparison if multiple years were analyzed
    if len(results) > 1:
        # Sort by year
        results.sort(key=lambda x: x['year'])
        
        years = [r['year'] for r in results]
        tourism_percentages = [r['tourism'] / r['total'] * 100 for r in results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(years, tourism_percentages)
        plt.title('Percentage of Tourism-Related Comments by Year')
        plt.ylabel('Percentage (%)')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig('visualizations/yearly_comparison.png')
        plt.close()
        
        print("\nYear-by-Year Comparison:")
        for r in results:
            print(f"{r['year']}: {r['tourism']} tourism-related comments out of {r['total']} ({r['tourism']/r['total']*100:.2f}%)")

if __name__ == "__main__":
    main() 