import pandas as pd
import os
import argparse
from datetime import datetime

def split_csv_by_year(input_file, date_column, output_dir='yearly_data', date_format=None):
    """
    Split a CSV file into multiple files based on the year in a date column.
    
    Args:
        input_file (str): Path to the input CSV file
        date_column (str): Name of the column containing date information
        output_dir (str): Directory to save the output files
        date_format (str, optional): Format of the date in the column
    """
    print(f"Reading {input_file}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV in chunks to handle large files
    chunk_size = 100000  # Adjust based on available memory
    
    # Dictionary to store dataframes by year
    year_dfs = {}
    
    # Process file in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Convert date column to datetime if format is provided
        if date_format:
            chunk[date_column] = pd.to_datetime(chunk[date_column], format=date_format)
        else:
            chunk[date_column] = pd.to_datetime(chunk[date_column], errors='coerce')
        
        # Extract year and split data
        chunk['year'] = chunk[date_column].dt.year
        
        # Group by year and append to corresponding dataframes
        for year, group in chunk.groupby('year'):
            if year not in year_dfs:
                year_dfs[year] = []
            
            # Drop the temporary year column
            group = group.drop('year', axis=1)
            year_dfs[year].append(group)
    
    # Concatenate chunks and save to files
    for year, chunks in year_dfs.items():
        if pd.isna(year):
            continue  # Skip rows with invalid dates
            
        year = int(year)
        print(f"Saving data for year {year}...")
        
        df = pd.concat(chunks)
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_{year}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split CSV file by year')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('date_column', help='Column name containing date information')
    parser.add_argument('--output_dir', default='yearly_data', help='Output directory for the split files')
    parser.add_argument('--date_format', help='Format of the date (e.g., "%%Y-%%m-%%d")')
    
    args = parser.parse_args()
    
    split_csv_by_year(
        args.input_file,
        args.date_column,
        args.output_dir,
        args.date_format
    ) 