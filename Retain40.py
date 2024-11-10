import os
import shutil

# Define the base directory
base_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/download_till_today'

# Function to count files in a directory
def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# Iterate over each subfolder in the base directory
for subfolder_name in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder_name)
    
    if os.path.isdir(subfolder_path):
        # Count the number of files in the subfolder
        file_count = count_files(subfolder_path)
        
        # Check if the file count is less than or equal to 40
        if file_count < 40:
            # Print the name of the subfolder that will be removed
            print(f"Removing {subfolder_path} with {file_count} files")
            
            # Remove the subfolder
            shutil.rmtree(subfolder_path)
        else:
            print(f"{subfolder_path} has more than 40 files, keeping it.")
