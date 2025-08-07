import re
import emoji
import json
import csv
import os
from nltk.tokenize import word_tokenize
import nltk
from googleapiclient.discovery import build
import ssl
from datetime import datetime

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

def preprocess_comment(comment):
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
    comment = emoji.demojize(comment, delimiters=("", " "))
    comment = re.sub(r'[^\w\s]', '', comment)
    tokens = word_tokenize(comment.lower())
    return ' '.join(tokens)

api_keys = ['AIzaSyC36Q9PYBNBm8dWJjsCEKBdqf6JtCpCxd4', 
        'AIzaSyBmnHd7qytvDVi4GT7_QRcDuunirppQ3y0',
        'AIzaSyCstRN73eNS2DRJB1jOhNdOdNd7s7qHMdg',
        'AIzaSyDUz6-wX5Zc2KFhDqc1AUngpHdHPNNMkVw',
        'AIzaSyA1VKhE_7eWUq9mazhLrJOt4bkcYfoNENY',
        'AIzaSyCNdiByBt6pso4RkctvTXKmIzm_Mr1a9S0',
        'AIzaSyDbKUHKGt5UFnPqzQkkgqgV3F_QSnwDf-s',
        'AIzaSyCdtBzIJJV6G7dR4sdeetTQuGsaDQKtufU',
        'AIzaSyDpbRRc7wiXCFxkxVuA8HWrKvufHX0-EPo',
        'AIzaSyDyzBAV5emQojvgysaZSviB1CF5cslxfzo',
        'AIzaSyAxnf4BITwQ5L7EyQ-ESYiRT63MzrY39wM',
        'IzaSyDEQAGSz2VMt4lht5uDUiZg1hC-bYcmvbQ']


def load_progress():
    try:
        with open("/arf/home/tbag175/progress.json", "r") as file:
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

    filename = "/arf/home/tbag175/youtube_comments.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        'comment_id', 'comment', 'author_name', 'author_channel_id',
        'author_channel_url', 'profile_image_url', 'like_count',
        'published_at', 'published_at_unix', 'updated_at',
        'updated_at_unix', 'reply_count', 'is_reply'
    ]

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for api_key in api_keys:
            try:
                while comment_count < comment_limit_per_key:
                    response = fetch_comments(api_key, video_id, max_results=50, next_page_token=next_page_token)

                    for item in response.get('items', []):
                        comment_data = item['snippet']['topLevelComment']['snippet']
                        comment_id = item['snippet']['topLevelComment']['id']
                        comment = comment_data['textDisplay']
                        author_name = comment_data.get('authorDisplayName', 'Unknown')
                        author_channel_id = comment_data.get('authorChannelId', {}).get('value', 'Unknown')
                        author_channel_url = comment_data.get('authorChannelUrl', 'Unknown')
                        profile_image_url = comment_data.get('authorProfileImageUrl', 'Unknown')
                        like_count = comment_data.get('likeCount', 0)
                        published_at = comment_data.get('publishedAt', 'Unknown')
                        updated_at = comment_data.get('updatedAt', 'Unknown')
                        reply_count = item['snippet']['totalReplyCount']
                        is_reply = 'parentId' in item['snippet']

                        # Convert timestamps to Unix time
                        try:
                            published_at_unix = int(datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").timestamp())
                            updated_at_unix = int(datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ").timestamp()) if updated_at != 'Unknown' else 'Unknown'
                        except ValueError:
                            published_at_unix = 'Unknown'
                            updated_at_unix = 'Unknown'

                        writer.writerow({
                            'comment_id': comment_id,
                            'comment': preprocess_comment(comment),
                            'author_name': author_name,
                            'author_channel_id': author_channel_id,
                            'author_channel_url': author_channel_url,
                            'profile_image_url': profile_image_url,
                            'like_count': like_count,
                            'published_at': published_at,
                            'published_at_unix': published_at_unix,
                            'updated_at': updated_at,
                            'updated_at_unix': updated_at_unix,
                            'reply_count': reply_count,
                            'is_reply': is_reply
                        })
                        comment_count += 1

                        print({
                            'comment_id': comment_id,
                            'comment': preprocess_comment(comment),
                            'author_name': author_name,
                            'author_channel_id': author_channel_id,
                            'author_channel_url': author_channel_url,
                            'profile_image_url': profile_image_url,
                            'like_count': like_count,
                            'published_at': published_at,
                            'published_at_unix': published_at_unix,
                            'updated_at': updated_at,
                            'updated_at_unix': updated_at_unix,
                            'reply_count': reply_count,
                            'is_reply': is_reply
                        })

                        if comment_count >= comment_limit_per_key:
                            break

                    next_page_token = response.get("nextPageToken")

                    save_progress(comment_count, next_page_token)

                    if not next_page_token:
                        break

            except Exception as e:
                print(f"Error with API key {api_key}: {e}")

gather_comments(video_id='kJQP7kiw5Fk')
print("Comments saved to youtube_comments.csv")

