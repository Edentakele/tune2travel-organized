import os
import pandas as pd
import argparse
from tqdm import tqdm

# Define the tourism/destination marketing keywords
TOURISM_KEYWORDS = [
    'puerto rico', 'visit', 'vacation', 'tourist', 'tourism', 'travel', 
    'destination', 'beach', 'island', 'resort', 'hotel', 'trip', 'visit puerto rico',
    'beautiful place', 'san juan', 'caribbean', 'beautiful country', 'paradise',
    'place to visit', 'vacation spot', 'landscape', 'scenery', 'venue', 'location',
    'la perla', 'old san juan'
]

def is_tourism_related(comment):
    """
    Check if a comment is related to tourism or destination marketing.
    Returns True for tourism-related, False for not related.
    """
    # Convert to lowercase for case-insensitive matching
    if not isinstance(comment, str):
        return False
        
    comment_lower = comment.lower()
    
    # Check for tourism keywords
    for keyword in TOURISM_KEYWORDS:
        if keyword in comment_lower:
            return True
    
    return False

def process_file(input_file, output_file, sample_size=None, start_row=None, end_row=None):
    """Process a CSV file and label comments."""
    print(f"Processing {input_file}...")
    
    # Read the CSV file (with sampling or row range if specified)
    try:
        # Get total rows for information purposes
        total_rows = sum(1 for _ in open(input_file)) - 1
        print(f"Total rows in file: {total_rows}")
        
        if start_row is not None and end_row is not None:
            # Process a specific range of rows
            skiprows = start_row
            nrows = end_row - start_row
            print(f"Processing rows {start_row} to {end_row} (total: {nrows} rows)...")
            df = pd.read_csv(input_file, skiprows=skiprows, nrows=nrows)
            if skiprows > 0:
                # Re-add the header
                header_df = pd.read_csv(input_file, nrows=1)
                df.columns = header_df.columns
        elif sample_size:
            # Take a random sample
            sample_size = min(sample_size, total_rows)
            print(f"Taking a sample of approximately {sample_size} rows...")
            df = pd.read_csv(input_file)
            df = df.sample(n=sample_size, random_state=42)
        else:
            # Process the entire file
            df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Processing {len(df)} comments...")
    
    # Apply the labeling function
    df['tourism_related'] = df['comment'].apply(is_tourism_related).astype(int)
    
    # Save the labeled data
    df.to_csv(output_file, index=False)
    print(f"Labeled data saved to {output_file}")
    
    # Print summary statistics
    tourism_count = df['tourism_related'].sum()
    print(f"Tourism-related comments: {tourism_count} ({tourism_count/len(df)*100:.2f}%)")
    print(f"Non-tourism-related comments: {len(df) - tourism_count} ({(len(df) - tourism_count)/len(df)*100:.2f}%)")

def manually_review_sample(labeled_file, output_file, sample_size=100):
    """Manually review a sample of the labeled data."""
    if not os.path.exists(labeled_file):
        print(f"Labeled file {labeled_file} not found.")
        return
        
    df = pd.read_csv(labeled_file)
    
    # Take a random sample
    sample = df.sample(min(sample_size, len(df)))
    
    # Review the sample
    corrections = 0
    
    print(f"\nManually reviewing {len(sample)} comments...")
    
    for idx, row in sample.iterrows():
        comment = row['comment']
        auto_label = row['tourism_related']
        
        print("\n" + "="*80)
        print(f"Comment: {comment}")
        print(f"Automatic label: {'Tourism-related' if auto_label == 1 else 'Not tourism-related'}")
        
        while True:
            response = input("Is this correct? (y/n/q to quit): ").strip().lower()
            if response == 'q':
                break
                
            if response in ('y', 'n'):
                if response == 'n':
                    # Flip the label
                    df.at[idx, 'tourism_related'] = 1 - auto_label
                    corrections += 1
                break
                
            print("Invalid input. Please enter 'y' for yes, 'n' for no, or 'q' to quit.")
    
    if corrections > 0:
        # Save the corrected data
        df.to_csv(output_file, index=False)
        print(f"\nCorrected {corrections} labels. Updated data saved to {output_file}")
        print(f"Accuracy of automatic labeling: {(1 - corrections/len(sample))*100:.2f}%")
    else:
        print("\nNo corrections made.")

def main():
    parser = argparse.ArgumentParser(description='Label YouTube comments for tourism/destination marketing relevance')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--sample', '-s', type=int, help='Number of comments to sample (for large files)')
    parser.add_argument('--start-row', type=int, help='Start row index (0-based, excluding header)')
    parser.add_argument('--end-row', type=int, help='End row index (0-based, excluding header)')
    parser.add_argument('--review', '-r', action='store_true', help='Manually review a sample of automatic labels')
    parser.add_argument('--review-size', type=int, default=100, help='Number of comments to manually review')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sample and (args.start_row is not None or args.end_row is not None):
        print("Error: Cannot use both --sample and row range options (--start-row/--end-row)")
        return
    
    if (args.start_row is not None and args.end_row is None) or (args.start_row is None and args.end_row is not None):
        print("Error: Both --start-row and --end-row must be specified together")
        return
    
    if args.start_row is not None and args.end_row is not None and args.start_row >= args.end_row:
        print("Error: --start-row must be less than --end-row")
        return
    
    # Set default output filename if not provided
    if not args.output:
        base_name = os.path.basename(args.input)
        if args.start_row is not None and args.end_row is not None:
            args.output = f"labeled_{args.start_row}_{args.end_row}_{base_name}"
        else:
            args.output = f"labeled_{base_name}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    process_file(args.input, args.output, args.sample, args.start_row, args.end_row)
    
    # Manually review if requested
    if args.review:
        review_output = f"reviewed_{os.path.basename(args.output)}"
        manually_review_sample(args.output, review_output, args.review_size)

if __name__ == "__main__":
    main() 