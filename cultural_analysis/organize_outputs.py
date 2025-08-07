import os
import shutil
import glob

def organize_files():
    """
    Organizes output files from the cultural analysis into video-specific subdirectories.
    """
    # Define the base directory where the script is located.
    # We assume the output files are in a 'cultural_analysis' subdirectory.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir)

    if not os.path.isdir(source_dir):
        print(f"Error: The source directory '{source_dir}' does not exist. Please run this script from the project root.")
        return

    # Define prefixes and their corresponding destination folders
    prefixes = {
        'despa': os.path.join(source_dir, 'despa_outputs'),
        'seeyou': os.path.join(source_dir, 'seeyou_outputs')
    }

    # Create destination directories if they don't exist
    for folder_path in prefixes.values():
        os.makedirs(folder_path, exist_ok=True)
        print(f"Ensured directory exists: {folder_path}")

    # Define the file patterns to look for (CSVs and HTML files)
    file_patterns = ['*.csv', '*.html']
    
    # Iterate over the prefixes and move files
    for prefix, dest_folder in prefixes.items():
        print(f"\nProcessing files for prefix: '{prefix}'...")
        for pattern in file_patterns:
            # Create the full search pattern for the source directory
            search_pattern = os.path.join(source_dir, f'{prefix}_*{pattern}')
            
            # Find all files matching the pattern
            for file_path in glob.glob(search_pattern):
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(dest_folder, file_name)
                    
                    try:
                        shutil.move(file_path, dest_path)
                        print(f"Moved: {file_name} -> {dest_folder}")
                    except Exception as e:
                        print(f"Error moving {file_name}: {e}")

    print("\nFile organization complete.")

if __name__ == '__main__':
    organize_files() 