import pandas as pd

# Step 1: Load the CSV
df = pd.read_csv("youtube_comments_historical.csv")  # Replace with your filename

# Step 2: Convert 'published_at' to datetime format
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

# Step 3: Extract year into a new column
df['year'] = df['published_at'].dt.year

# Step 4: Group by year and save each year to a new CSV
for year, group in df.groupby('year'):
    group.to_csv(f'comments_{year}.csv', index=False)
    print(f"Saved: comments_{year}.csv with {len(group)} rows")
