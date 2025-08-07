import pandas as pd
from collections import Counter
import emoji
import os

# ✅ Unicode skin tone modifiers
SKIN_TONES = ['🏻', '🏼', '🏽', '🏾', '🏿']

# ✅ Function to extract and normalize emojis
def extract_and_normalize_emojis(text):
    normalized = []
    for char in str(text):
        if emoji.is_emoji(char):
            for tone in SKIN_TONES:
                char = char.replace(tone, '')  # Remove skin tone modifier
            normalized.append(char)
    return normalized

# ✅ Function to process one CSV file
def process_emoji_file(input_csv, output_csv):
    print(f"Processing {input_csv}...")
    df = pd.read_csv(input_csv)

    emoji_counter = Counter()

    for row in df['emoji_only'].dropna():
        emoji_counter.update(extract_and_normalize_emojis(row))

    emoji_df = pd.DataFrame(emoji_counter.items(), columns=["emoji", "count"])
    emoji_df = emoji_df.sort_values(by="count", ascending=False)
    emoji_df.to_csv(output_csv, index=False)
    print(f"✅ Done! Saved to {output_csv}\n")

# ✅ Process files from cleaned folder
base_path = "F:/extraction/Whole/comments with emoji splitted yearly/cleaned and yearly"

for year in range(2016, 2026):
    input_file = os.path.join(base_path, f"cleaned_{year}.csv")
    output_file = os.path.join(base_path, f"emoji_frequency_{year}.csv")
    try:
        process_emoji_file(input_file, output_file)
    except FileNotFoundError:
        print(f"⚠️ File {input_file} not found. Skipping...\n")
