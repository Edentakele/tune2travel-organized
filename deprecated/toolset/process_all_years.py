import os
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Process all years of Despacito comments and analyze tourism marketing trends')
    parser.add_argument('--sample', '-s', type=int, default=10000, help='Number of comments to sample per year')
    parser.add_argument('--output-dir', '-o', default='labeled_data', help='Directory to save labeled data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all years from the filtered_yearly_data folder
    years = []
    for filename in os.listdir('filtered_yearly_data'):
        if filename.endswith('.csv'):
            year = filename.split('_')[-1].split('.')[0]
            years.append(year)
    
    # Sort years
    years.sort()
    
    # Process each year
    results = []
    for year in tqdm(years, desc="Processing years"):
        input_file = f"filtered_yearly_data/filtered_despacito_{year}.csv"
        output_file = f"{args.output_dir}/labeled_{year}.csv"
        
        # Run the labeling script
        print(f"\nProcessing {year}...")
        subprocess.run([
            "python", "label_single_file.py", 
            "-i", input_file, 
            "-o", output_file,
            "-s", str(args.sample)
        ])
        
        # Read the results
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            tourism_count = df['tourism_related'].sum()
            total_count = len(df)
            
            results.append({
                'year': year,
                'tourism_count': tourism_count,
                'total_count': total_count,
                'percentage': (tourism_count / total_count) * 100
            })
    
    # Create a summary CSV
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(f"{args.output_dir}/summary_by_year.csv", index=False)
        print(f"\nSummary saved to {args.output_dir}/summary_by_year.csv")
        
        # Create summary visualizations
        os.makedirs('visualizations', exist_ok=True)
        
        # Tourism percentage by year
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['year'], summary_df['percentage'])
        plt.title('Percentage of Tourism-Related Comments by Year')
        plt.ylabel('Percentage (%)')
        plt.xlabel('Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/tourism_percentage_by_year.png')
        plt.close()
        
        # Tourism count by year
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['year'], summary_df['tourism_count'])
        plt.title('Number of Tourism-Related Comments by Year')
        plt.ylabel('Count')
        plt.xlabel('Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/tourism_count_by_year.png')
        plt.close()
        
        # Run the full visualization script on the output directory
        print("\nGenerating detailed visualizations...")
        subprocess.run(["python", "visualize_results.py", "-d", args.output_dir])
        
        # Print summary
        print("\nSummary of Tourism-Related Comments by Year:")
        for row in results:
            print(f"{row['year']}: {row['tourism_count']} out of {row['total_count']} comments ({row['percentage']:.2f}%)")

if __name__ == "__main__":
    main() 