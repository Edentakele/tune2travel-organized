import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder where the emoji_frequency_20XX.csv files are stored
base_path = "F:/extraction/Whole/comments with emoji splitted yearly/cleaned and yearly"

for year in range(2016, 2026):
    input_file = os.path.join(base_path, f"emoji_frequency_{year}.csv")
    output_file = os.path.join(base_path, f"emoji_bar_top15_{year}.png")

    try:
        df = pd.read_csv(input_file)

        # Clean and limit
        df = df.dropna(subset=['emoji'])
        df['emoji'] = df['emoji'].astype(str)
        df = df[df['count'] >= 1]  # Can lower if needed
        df = df.sort_values(by='count', ascending=False).head(15)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(df['emoji'], df['count'], color='skyblue')
        plt.xlabel("Emoji")
        plt.ylabel("Frequency")
        plt.title(f"Top 15 Emojis in {year}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"✅ Saved bar chart for {year} ➜ {output_file}")

    except FileNotFoundError:
        print(f"⚠️ File not found: {input_file} — skipping...\n")
    except Exception as e:
        print(f"❌ Error processing {year}: {str(e)}\n")
