import re
import emoji
import json
import csv
import os
from nltk.tokenize import word_tokenize
import nltk
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

DetectorFactory.seed = 0
nltk.download('punkt')

KEYWORDS = r'\b(?:travel|experience|visit|city|adventure|sightseeing|destination|explore|vacation|trip|tour|journey|summer|advise|holiday|tourist|food)\b'

def preprocess_comment(comment):
    try:
        if detect(comment) != 'en':
            return None  
    except Exception as e:
        return None  

    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)    
    comment = emoji.demojize(comment, delimiters=("", " "))
    comment = re.sub(r'[^\w\s]', '', comment)

    if not re.search(KEYWORDS, comment, re.IGNORECASE):
        return None  

    tokens = word_tokenize(comment.lower())
    return ' '.join(tokens)

comments_dataset = []

api_keys = ['AIzaSyC36Q9PYBNBm8dWJjsCEKBdqf6JtCpCxd4','AIzaSyBmnHd7qytvDVi4GT7_QRcDuunirppQ3y0']  

def load_progress():
    try:
        with open("progress.json", "r") as file:
            progress = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {"comment_count": 0, "next_page_token": None}
    return progress

def save_progress(comment_count, next_page_token):
    with open("progress.json", "w") as file:
        json.dump({"comment_count": comment_count, "next_page_token": next_page_token}, file)

def fetch_comments(api_key, video_id, max_results, next_page_token=None):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        pageToken=next_page_token
    )
    response = request.execute()
    return response

def gather_comments(video_id, comment_limit_per_key=1000000000):
    progress = load_progress()
    comment_count = progress.get("comment_count", 0)
    next_page_token = progress.get("next_page_token", None)

    for api_key in api_keys:
        try:
            while comment_count < comment_limit_per_key:
                response = fetch_comments(api_key, video_id, max_results=50, next_page_token=next_page_token)
                
                for item in response.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    processed_comment = preprocess_comment(comment)
                    
                    if processed_comment:  
                        comments_dataset.append({
                            'comment': processed_comment
                        })
                        comment_count += 1
                    
                    if comment_count >= comment_limit_per_key:
                        break
                
                next_page_token = response.get("nextPageToken")
                
                save_progress(comment_count, next_page_token)
                
                if not next_page_token:
                    break

        except Exception as e:
            print(f"Error with API key {api_key}: {e}")

def save_to_csv(filename="youtube_comments.csv"):
    file_exists = os.path.isfile(filename)
    fieldnames = ['comment']

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(comments_dataset)

gather_comments(video_id='kJQP7kiw5Fk')  
save_to_csv("youtube_comments.csv")
print(f"Total comments collected: {len(comments_dataset)}")
print("Comments saved to youtube_comments.csv")
