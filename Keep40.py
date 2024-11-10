import os

# Define the base directory containing subfolders
base_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/data_9thOctManual_croppedUpdated'

# Function to count the number of files in a directory
def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# Function to keep only 'n' files in a directory and remove the rest
def keep_only_n_files(directory, n):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if len(files) > n:
        # Sort files by name to ensure consistent ordering (optional)
        files.sort()
        # Determine which files to delete
        files_to_delete = files[n:]
        for file_name in files_to_delete:
            file_path = os.path.join(directory, file_name)
            print(f"Removing {file_path}")
            os.remove(file_path)

# Iterate over each subfolder in the base directory
for subfolder_name in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder_name)
    
    if os.path.isdir(subfolder_path):
        # Count the number of files in the subfolder
        file_count = count_files(subfolder_path)
        
        if file_count > 40:
            print(f"Subfolder {subfolder_path} has {file_count} files, trimming to 40 files.")
            # Remove extra files
            keep_only_n_files(subfolder_path, 40)
        else:
            print(f"Subfolder {subfolder_path} has {file_count} files, no changes needed.")
