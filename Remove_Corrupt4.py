from PIL import Image
import os

def check_and_list_corrupt_images(directory):
    corrupt_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()  # Verify the image
                except (IOError, SyntaxError) as e:
                    print(f'Corrupt file detected: {filepath} - {e}')
                    corrupt_files.append(filepath)
    return corrupt_files

def remove_files(filepaths):
    for filepath in filepaths:
        try:
            os.remove(filepath)
            print(f'Removed file: {filepath}')
        except Exception as e:
            print(f'Error removing file {filepath}: {e}')

def main():
    train_dir = "/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/V14_train184"
    #val_dir = "path/to/validation/directory"

    # Check for corrupt images
    corrupt_train_files = check_and_list_corrupt_images(train_dir)
    #corrupt_val_files = check_and_list_corrupt_images(val_dir)

    print(f'Corrupt training files: {corrupt_train_files}')
    #print(f'Corrupt validation files: {corrupt_val_files}')

    # Remove corrupt files
    remove_files(corrupt_train_files)
    #remove_files(corrupt_val_files)

if __name__ == "__main__":
    main()
