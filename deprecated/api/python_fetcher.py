from googleapiclient.discovery import build

# Replace with your own API key
API_KEY = 'YOUR_API_KEY'
VIDEO_ID = 'YOUR_VIDEO_ID'

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(video_id):
    comments = []
    # Call the API to get comments
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100  # Maximum number of comments per request
    ).execute()

    # Loop through the response and extract comments
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        # Check if there are more comments to fetch
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                textFormat='plainText',
                maxResults=100
            ).execute()
        else:
            break

    return comments

# Fetch comments
comments = get_comments(VIDEO_ID)
for comment in comments:
    print(comment)
