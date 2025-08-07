import re
import emoji
import json
import csv
import os
from datetime import datetime

# Define the date range from today to 2016
END_DATE = datetime.now()
START_DATE = datetime(2016, 1, 1)
START_TIMESTAMP = int(START_DATE.timestamp())
END_TIMESTAMP = int(END_DATE.timestamp())

print(f"Collecting comments from {END_DATE.strftime('%Y-%m-%d')} back to {START_DATE.strftime('%Y-%m-%d')}")

def extract_emojis(text):
    return ''.join(char for char in text if char in emoji.EMOJI_DATA)

def preprocess_comment(comment):
    emojis = extract_emojis(comment)
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
    comment = re.sub(r'[^\w\s]', '', comment)
    words = comment.lower().split()
    cleaned_comment = ' '.join(words)
    return cleaned_comment, emojis

api_keys = [
    
    'AIzaSyAjKMn_StNBd7RVAH-OSchrgQVaHvghLPY',
    'AIzaSyCpqIxWGTF7-Ul15utcu7TdrIyAoJAoXis',
    'AIzaSyAMWzEF_RGGIJ3FjheiFxh2Oo-aGtOTFaE'

    
]

def load_progress():
    try:
        with open("./progress2.json", "r") as file:
            progress = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {"comment_count": 0, "next_page_token": None, "filtered_count": 0}
    return progress

def save_progress(comment_count, next_page_token, filtered_count):
    with open("progress2.json", "w") as file:
        json.dump({
            "comment_count": comment_count, 
            "next_page_token": next_page_token,
            "filtered_count": filtered_count
        }, file)

def fetch_comments(api_key, video_id, max_results, next_page_token=None):
    try:
        from googleapiclient.discovery import build
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token
        )
        response = request.execute()
        return response
    except Exception as e:
        print(f"Error fetching comments: {str(e)}")
        return None

def gather_comments(video_id, comment_limit_per_key=1000000000):
    if not api_keys or api_keys[0] == 'YOUR_API_KEY_1':
        print("Error: Please replace the placeholder API keys with valid YouTube Data API keys")
        return 0

    progress = load_progress()
    comment_count = progress.get("comment_count", 0)
    filtered_count = progress.get("filtered_count", 0)
    next_page_token = progress.get("next_page_token", None)

    filename = "./youtube_comments_historical.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        'comment_id', 'comment', 'emoji_only', 'author_name', 'author_channel_id',
        'author_channel_url', 'profile_image_url', 'like_count',
        'published_at', 'published_at_unix', 'updated_at',
        'updated_at_unix', 'reply_count', 'is_reply'
    ]

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for api_key in api_keys:
            print(f"\nTrying API key: {api_key[:8]}...")
            try:
                while filtered_count < comment_limit_per_key:
                    response = fetch_comments(api_key, video_id, max_results=50, next_page_token=next_page_token)

                    if not response:
                        print(f"No response from API with key {api_key[:8]}... Trying next key.")
                        break

                    items = response.get('items', [])
                    if not items:
                        print("No comments found in response")
                        break

                    for item in items:
                        comment_data = item['snippet']['topLevelComment']['snippet']
                        published_at = comment_data.get('publishedAt', 'Unknown')

                        try:
                            published_at_unix = int(datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").timestamp())

                            if START_TIMESTAMP <= published_at_unix <= END_TIMESTAMP:
                                comment_id = item['snippet']['topLevelComment']['id']
                                original_comment = comment_data['textDisplay']
                                cleaned_comment, emoji_only = preprocess_comment(original_comment)
                                author_name = comment_data.get('authorDisplayName', 'Unknown')
                                author_channel_id = comment_data.get('authorChannelId', {}).get('value', 'Unknown')
                                author_channel_url = comment_data.get('authorChannelUrl', 'Unknown')
                                profile_image_url = comment_data.get('authorProfileImageUrl', 'Unknown')
                                like_count = comment_data.get('likeCount', 0)
                                updated_at = comment_data.get('updatedAt', 'Unknown')
                                reply_count = item['snippet']['totalReplyCount']
                                is_reply = 'parentId' in item['snippet']

                                try:
                                    updated_at_unix = int(datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ").timestamp()) if updated_at != 'Unknown' else 'Unknown'
                                except ValueError:
                                    updated_at_unix = 'Unknown'

                                writer.writerow({
                                    'comment_id': comment_id,
                                    'comment': cleaned_comment,
                                    'emoji_only': emoji_only,
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

                                filtered_count += 1
                                if filtered_count % 100 == 0:
                                    print(f"[{filtered_count}] Comments saved. Latest from: {published_at[:10]}")

                                if filtered_count >= comment_limit_per_key:
                                    break
                        except ValueError as ve:
                            print(f"Error processing date: {ve}")
                            continue

                        comment_count += 1

                        if comment_count % 1000 == 0:
                            print(f"Processed {comment_count} total comments, saved {filtered_count} comments from {START_DATE.year} to present")
                            save_progress(comment_count, next_page_token, filtered_count)

                    next_page_token = response.get("nextPageToken")
                    save_progress(comment_count, next_page_token, filtered_count)

                    if not next_page_token:
                        print("No more comments to fetch")
                        break

            except Exception as e:
                print(f"Error with API key {api_key[:8]}...: {str(e)}")
                continue

    print(f"\nCollection complete: Processed {comment_count} comments, saved {filtered_count} comments from {START_DATE.year} to present")
    return filtered_count

if __name__ == "__main__":
    video_id = 'RgKAFK5djSk'  # Despacito video ID
    total_saved = gather_comments(video_id)
    print(f"Total comments saved: {total_saved}")
    print(f"Comments saved to youtube_comments_historical.csv")
