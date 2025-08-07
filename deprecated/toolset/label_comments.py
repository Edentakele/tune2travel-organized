import os
import pandas as pd
import csv
from tqdm import tqdm
import sys

# Define the tourism/destination marketing keywords
TOURISM_KEYWORDS = [
    'puerto rico', 'visit', 'vacation', 'tourist', 'tourism', 'travel', 
    'destination', 'beach', 'island', 'resort', 'hotel', 'trip', 'visit puerto rico',
    'beautiful place', 'san juan', 'caribbean', 'beautiful country', 'paradise',
    'place to visit', 'vacation spot', 'landscape', 'scenery', 'venue', 'location'
]

def is_tourism_related(comment):
    """
    Check if a comment is related to tourism or destination marketing.
    Returns 1 for tourism-related, 0 for not related, -1 for manual review.
    """
    # Convert to lowercase for case-insensitive matching
    comment_lower = comment.lower()
    
    # Check for tourism keywords
    for keyword in TOURISM_KEYWORDS:
        if keyword in comment_lower:
            return 1
    
    # If no clear indicators, mark for manual review
    return -1

def process_file(input_file, output_file, manual_review_file, batch_size=1000):
    """Process a CSV file and label comments."""
    print(f"Processing {input_file}...")
    
    # Read file in chunks to handle large files
    labeled_data = []
    manual_review_data = []
    
    for chunk in tqdm(pd.read_csv(input_file, chunksize=batch_size)):
        for _, row in chunk.iterrows():
            comment = str(row['comment'])
            label = is_tourism_related(comment)
            
            # Add to appropriate output list
            if label == 1:  # Tourism related
                labeled_data.append({**row.to_dict(), 'tourism_related': 1})
            elif label == 0:  # Not tourism related
                labeled_data.append({**row.to_dict(), 'tourism_related': 0})
            else:  # Manual review
                manual_review_data.append(row.to_dict())
    
    # Save labeled data
    if labeled_data:
        pd.DataFrame(labeled_data).to_csv(output_file, index=False)
        print(f"Labeled data saved to {output_file}")
    
    # Save data for manual review
    if manual_review_data:
        pd.DataFrame(manual_review_data).to_csv(manual_review_file, index=False)
        print(f"Data for manual review saved to {manual_review_file}")

def manual_labeling(manual_review_file, output_file):
    """Manually label comments that couldn't be automatically classified."""
    if not os.path.exists(manual_review_file):
        print(f"Manual review file {manual_review_file} not found.")
        return
    
    data = pd.read_csv(manual_review_file)
    if data.empty:
        print("No comments to manually label.")
        return
    
    labeled_data = []
    
    print(f"Manual labeling: {len(data)} comments to review")
    print("For each comment, enter 1 for tourism-related, 0 for not tourism-related, or q to quit")
    
    for idx, row in data.iterrows():
        comment = row['comment']
        print("\n" + "="*80)
        print(f"Comment {idx+1}/{len(data)}: {comment}")
        
        while True:
            response = input("Tourism-related? (1=Yes, 0=No, q=Quit): ").strip().lower()
            if response == 'q':
                # Save progress and exit
                if labeled_data:
                    pd.DataFrame(labeled_data).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                    print(f"Progress saved to {output_file}")
                return
            
            if response in ('0', '1'):
                label = int(response)
                labeled_data.append({**row.to_dict(), 'tourism_related': label})
                break
            
            print("Invalid input. Please enter 1 for tourism-related, 0 for not tourism-related, or q to quit.")
    
    # Save all manually labeled data
    if labeled_data:
        pd.DataFrame(labeled_data).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        print(f"All manually labeled data saved to {output_file}")

def main():
    # Create output directories if they don't exist
    os.makedirs('labeled_data', exist_ok=True)
    os.makedirs('manual_review', exist_ok=True)
    os.makedirs('final_labeled', exist_ok=True)
    
    # Process all CSV files in the filtered_yearly_data folder
    data_dir = 'filtered_yearly_data'
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(data_dir, filename)
            output_file = os.path.join('labeled_data', f'auto_{filename}')
            manual_review_file = os.path.join('manual_review', f'manual_{filename}')
            
            process_file(input_file, output_file, manual_review_file)
            
            # Manual labeling
            final_file = os.path.join('final_labeled', f'final_{filename}')
            manual_labeling(manual_review_file, final_file)

if __name__ == "__main__":
    main() 