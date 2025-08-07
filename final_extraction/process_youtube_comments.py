import re
import emoji
import json
import csv
import os
import pandas as pd
from langdetect import detect, LangDetectException
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Define the date range (optional, can be removed if not needed)
# END_DATE = datetime.now()
# START_DATE = datetime(2016, 1, 1)
# START_TIMESTAMP = int(START_DATE.timestamp())
# END_TIMESTAMP = int(END_DATE.timestamp())

# print(f"Collecting comments from {END_DATE.strftime('%Y-%m-%d')} back to {START_DATE.strftime('%Y-%m-%d')}")

# --- Helper Functions ---

def extract_emojis(text):
    """Extracts all emoji characters from a string."""
    return ''.join(char for char in text if char in emoji.EMOJI_DATA)

def remove_emojis(text):
    """Removes all emoji characters from a string."""
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA)

def get_emoji_text(text):
    """Converts emojis in a string to their text representation (e.g., :red_heart:)."""
    return emoji.demojize(text)

def detect_language(text):
    """Detects the language of a text snippet."""
    try:
        # Basic cleaning to improve language detection
        text_cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip() # Remove extra whitespace
        if not text_cleaned or len(text_cleaned) < 10: # Avoid detection on very short/empty strings
             return "und" # Undetermined
        return detect(text_cleaned)
    except LangDetectException:
        return "error" # Indicate detection failure
    except Exception as e:
        print(f"Unexpected error during language detection: {e}")
        return "error"

def process_comment_data(original_comment):
    """Processes a single comment to extract required fields."""
    original_comment_str = str(original_comment) # Ensure it's a string

    emojis = extract_emojis(original_comment_str)
    comment_no_emojis = remove_emojis(original_comment_str)
    emoji_text = get_emoji_text(emojis)
    language = detect_language(comment_no_emojis)

    # Optional: Further clean the comment_no_emojis text (e.g., remove URLs, special chars)
    # comment_no_emojis_cleaned = re.sub(r'http\S+|www\S+|https\S+', '', comment_no_emojis, flags=re.MULTILINE)
    # comment_no_emojis_cleaned = re.sub(r'[^\w\s]', '', comment_no_emojis_cleaned)
    # comment_no_emojis_cleaned = ' '.join(comment_no_emojis_cleaned.lower().split())

    return {
        "original_comment": original_comment_str,
        "comment_no_emojis": comment_no_emojis, # Or comment_no_emojis_cleaned
        "emojis_in_comment": emojis,
        "emoji_text_representation": emoji_text,
        "detected_language": language,
    }

# --- YouTube API Interaction ---

# Placeholder for API keys - replace with your actual keys or load securely
# It's recommended to load keys from environment variables or a config file
API_KEYS = [
    'AIzaSyAgF6ENHx_JZHCPUXGos0xZLiua3POLx_M',
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
    'AIzaSyDEQAGSz2VMt4lht5uDUiZg1hC-bYcmvbQ',
    'AIzaSyAjKMn_StNBd7RVAH-OSchrgQVaHvghLPY',
    'AIzaSyCpqIxWGTF7-Ul15utcu7TdrIyAoJAoXis',
    'AIzaSyAMWzEF_RGGIJ3FjheiFxh2Oo-aGtOTFaE'
]
CURRENT_API_KEY_INDEX = 0

def get_next_api_key():
    """Rotates through the available API keys."""
    global CURRENT_API_KEY_INDEX
    key = API_KEYS[CURRENT_API_KEY_INDEX]
    CURRENT_API_KEY_INDEX = (CURRENT_API_KEY_INDEX + 1) % len(API_KEYS)
    # Simple check for placeholder keys
    if 'YOUR_API_KEY' in key:
         print(f"[WARNING] Using placeholder API key: {key}. Replace it with a valid key.")
         # Optionally raise an error or return None if no valid keys are found
         # raise ValueError("No valid API keys configured.")
    return key

def fetch_comments_page(youtube, video_id, page_token=None):
    """Fetches a single page of comments for a video."""
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100, # Max allowed by API
            pageToken=page_token,
            order="time", # Use time order instead of relevance
        )
        response = request.execute()
        return response
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        if e.resp.status == 403: # Quota exceeded or access forbidden
            print("Quota likely exceeded for the current key or API access issue.")
            return None # Signal to try the next key
        elif e.resp.status == 404: # Video not found or comments disabled
             print(f"Video {video_id} not found or comments are disabled.")
             return {"items": [], "nextPageToken": None} # Treat as no comments
        else:
            raise # Re-raise other HTTP errors
    except Exception as e:
        print(f"Error fetching comments page: {str(e)}")
        raise # Re-raise other exceptions

# --- Progress Handling ---
PROGRESS_FILE = "comment_progress.json"

def load_progress(filename=PROGRESS_FILE):
    """Loads progress (nextToken, pageCount per video ID) from a JSON file."""
    try:
        with open(filename, 'r') as f:
            progress_data = json.load(f)
            # Basic validation/conversion from old format if necessary
            validated_progress = {}
            for video_id, data in progress_data.items():
                if isinstance(data, dict) and 'nextToken' in data and 'pageCount' in data:
                    validated_progress[video_id] = data
                elif isinstance(data, str) or data is None:
                    # Old format detected, convert it, assuming page count needs reset
                    print(f"  Converting old progress format for video {video_id}")
                    validated_progress[video_id] = {"nextToken": data, "pageCount": 0}
                else:
                    print(f"  Skipping unrecognized progress format for video {video_id}")

            print(f"Loaded and validated progress from {filename}")
            return validated_progress
    except FileNotFoundError:
        print("Progress file not found, starting fresh.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding progress file {filename}, starting fresh.")
        return {}
    except Exception as e:
        print(f"Error loading progress: {e}. Starting fresh.")
        return {}

def save_progress(progress_data, filename=PROGRESS_FILE):
    """Saves progress ({nextToken, pageCount} per video ID) to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(progress_data, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Failed to save progress to {filename}: {e}")

# --- Main Gathering Logic ---

def gather_and_process_video_comments(video_ids):
    """Gathers comments for multiple videos, processes them page by page,
    appends each page to a video-specific CSV file, and supports resuming."""
    total_comments_processed_all_videos = 0
    api_key_errors = 0
    progress = load_progress() # Load progress at the start

    # Define CSV fieldnames (ensure this matches data written)
    csv_fieldnames = [
        'comment_id', 'original_comment', 'comment_no_emojis',
        'emojis_in_comment', 'emoji_text_representation', 'detected_language',
        'author_name', 'published_at', 'published_at_unix', 'like_count'
    ]

    # Check if API keys are placeholders
    if not API_KEYS or all('YOUR_API_KEY' in k for k in API_KEYS if k):
        print("[ERROR] No valid YouTube API keys found or only placeholders present in API_KEYS list. Please add your keys.")
        return

    print(f"Starting comment gathering for video IDs: {', '.join(video_ids)}")

    for video_id in video_ids:
        print(f"\nProcessing video: {video_id}")

        # --- Resume Logic ---
        video_progress = progress.get(video_id, {})
        current_page_token = video_progress.get("nextToken", "START") # Default to START if no entry
        start_page_count = video_progress.get("pageCount", 0) # Default to 0

        if current_page_token is None and start_page_count > 0: # Explicitly check for None from previous completion
            print(f"  Video {video_id} already marked as completed (at page {start_page_count}) in progress file. Skipping.")
            continue
        elif current_page_token == "START":
             current_page_token = None
             page_count = 0 # Start count from 0
             print(f"  Starting fresh for video {video_id}.")
        else:
             page_count = start_page_count # Resume count from last saved page
             print(f"  Resuming video {video_id} from page {page_count + 1} (token: {current_page_token[:10]}...)")
        # --- End Resume Logic ---

        output_file_for_video = f"{video_id}_comments.csv" # Change extension
        video_comment_count = 0
        active_api_key = get_next_api_key()
        youtube = build('youtube', 'v3', developerKey=active_api_key)

        # --- CSV File Handling ---
        file_exists = os.path.exists(output_file_for_video)
        csv_file = None
        csv_writer = None
        try:
            # Open file in append mode, it will be kept open during page processing
            csv_file = open(output_file_for_video, 'a', newline='', encoding='utf-8')
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            if not file_exists:
                csv_writer.writeheader()
                print(f"  Initialized NEW CSV file: {output_file_for_video}")
            else:
                 print(f"  Appending to existing CSV file: {output_file_for_video}")

            # --- Page Fetching Loop ---
            while True:
                page_comments_data = [] # Data for the current page (list of dicts)
                # Increment page count *before* fetching the page
                current_processing_page_number = page_count + 1
                print(f"  Fetching page {current_processing_page_number} for video {video_id} (token: {current_page_token})...")
                response = fetch_comments_page(youtube, video_id, page_token=current_page_token)

                # Handle API key exhaustion/errors
                if response is None:
                    api_key_errors += 1
                    print(f"  Switching API key due to error with key ending in ...{active_api_key[-4:]} (Page {current_processing_page_number}, Token: {current_page_token})")
                    if api_key_errors >= len(API_KEYS):
                        print(f"  [ERROR] All API keys have encountered errors. Stopping collection for video {video_id}.")
                        break # Break inner loop
                    active_api_key = get_next_api_key()
                    youtube = build('youtube', 'v3', developerKey=active_api_key)
                    continue # Retry fetching the same page with the new key

                api_key_errors = 0 # Reset error count on successful fetch
                next_page_token = response.get("nextPageToken") # Get token for the *next* page

                items = response.get('items', [])
                if not items and current_page_token is None and video_comment_count == 0:
                     print(f"  No comments found or comments disabled for video {video_id}.")
                     break # Move to next video if no comments found on the first page

                # --- Process Items for the current page ---
                page_start_month_year = None
                page_end_month_year = None
                page_comments_processed_count = 0

                for item_index, item in enumerate(items):
                    try:
                        # Extract top-level comment
                        comment_snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                        original_comment = comment_snippet.get('textDisplay') # This now contains HTML if textFormat defaulted
                        published_at_str = comment_snippet.get('publishedAt')
                        comment_unix_ts = None # Initialize timestamp

                        if original_comment and published_at_str:
                            # Extract month-year for logging and calculate Unix timestamp
                            try:
                                comment_dt = datetime.strptime(published_at_str, "%Y-%m-%dT%H:%M:%SZ")
                                current_month_year = comment_dt.strftime("%Y-%m")
                                comment_unix_ts = int(comment_dt.timestamp()) # Calculate Unix timestamp
                                if item_index == 0: page_start_month_year = current_month_year
                                page_end_month_year = current_month_year
                            except ValueError: pass # Ignore if date parsing fails

                            processed_data = process_comment_data(original_comment) # Process comment (emoji, lang detect)

                            # Add other metadata
                            processed_data['comment_id'] = item.get('snippet', {}).get('topLevelComment', {}).get('id')
                            processed_data['author_name'] = comment_snippet.get('authorDisplayName')
                            processed_data['published_at'] = published_at_str
                            processed_data['published_at_unix'] = comment_unix_ts # Add Unix timestamp
                            processed_data['like_count'] = comment_snippet.get('likeCount')

                            # Ensure only defined fields are kept for CSV writing
                            filtered_data = {k: processed_data.get(k) for k in csv_fieldnames}
                            page_comments_data.append(filtered_data) # Add dict to page-specific list

                            video_comment_count += 1
                            total_comments_processed_all_videos += 1
                            page_comments_processed_count += 1

                    except Exception as e:
                        print(f"  [WARNING] Error processing individual comment item for video {video_id}: {e}. Skipping item.")
                        print(f"  Problematic item data: {item}")
                        continue # Skip to next item

                # --- Write current page data to CSV ---
                if page_comments_data and csv_writer:
                    try:
                        csv_writer.writerows(page_comments_data) # Write all rows for the page
                        print(f"  Appended {len(page_comments_data)} comments from page {current_processing_page_number} to {output_file_for_video}")

                        # --- Save Progress ---
                        actual_next_token = response.get("nextPageToken")
                        # Save the state needed to fetch the *next* page
                        progress[video_id] = {
                            "nextToken": actual_next_token,
                            "pageCount": current_processing_page_number # Save the number of the page just processed
                        }
                        save_progress(progress)
                        # --- End Save Progress ---

                    except Exception as e:
                         print(f"  [ERROR] Failed to write page {current_processing_page_number} data for video {video_id} to CSV: {e}")
                         # Decide whether to break or continue - continuing might lose data but allow other videos to process
                         break # Stop processing this video on write error

                # --- Log Page Summary ---
                month_year_range_str = ""
                if page_start_month_year and page_end_month_year:
                    if page_start_month_year == page_end_month_year: month_year_range_str = f"Month-Year: {page_start_month_year}"
                    else: month_year_range_str = f"Month-Years: {page_start_month_year} to {page_end_month_year}"
                elif page_start_month_year: month_year_range_str = f"Month-Year: {page_start_month_year}"

                print(f"  Page {current_processing_page_number} fetch complete ({page_comments_processed_count} comments processed). {month_year_range_str}. Video Total: {video_comment_count}")

                # --- Loop Control ---
                actual_next_token = response.get("nextPageToken")
                if not actual_next_token:
                    print(f"  Finished fetching all comments for video {video_id}. Total for video: {video_comment_count} (Processed {current_processing_page_number} pages)")
                    # Mark as complete in progress file
                    # progress[video_id] = {
                    #     "nextToken": None, 
                    #     "pageCount": current_processing_page_number
                    # }
                    # save_progress(progress) # DO NOT save the 'None' token state
                    break # Exit while loop for this video
                else:
                    current_page_token = actual_next_token
                    page_count = current_processing_page_number # Update page count for the next loop iteration

        # --- End Page Fetching Loop ---

        except HttpError as e:
                 print(f"  [ERROR] Unhandled HTTP error during pagination for video {video_id} (Page {current_processing_page_number}, Token: {current_page_token}): {e}. Stopping collection for this video.")
                 # Outer loop will continue to next video ID
        except Exception as e:
                print(f"  [ERROR] Unexpected error during processing loop for video {video_id} (Page {current_processing_page_number}, Token: {current_page_token}): {e}. Stopping video processing.")
                # Outer loop will continue to next video ID
        finally: # Ensure CSV file is closed
            if csv_file:
                try:
                    csv_file.close()
                    print(f"  Closed CSV file: {output_file_for_video}")
                except Exception as e:
                    print(f"  [ERROR] Failed to close CSV file {output_file_for_video}: {e}")
            # elif video_comment_count > 0 : # Optional: Log if comments were processed but file wasn't opened/written
            #      print(f"  [WARNING] No data was successfully written to {output_file_for_video} despite processing comments.")

    # Final summary message (inside the function)
    print(f"\nFinished processing all videos. Total comments processed across all videos: {total_comments_processed_all_videos}")

# --- Main Execution Starts Here ---

if __name__ == "__main__":
    # --- Configuration ---
    # Video IDs for "Despacito" and "See You Again"
    # Despacito: kJQP7kiw5Fk (Confirmed)
    # See You Again: RgKAFK5djSk (Confirmed)
    TARGET_VIDEO_IDS = ['kJQP7kiw5Fk', 'RgKAFK5djSk']

    # IMPORTANT: Replace placeholder API keys in the API_KEYS list near the top
    # of the script before running.

    print("Starting YouTube comment processing script...")
    # --- Run the process ---
    gather_and_process_video_comments(TARGET_VIDEO_IDS)
    print("Script finished.") 