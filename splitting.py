import os
import random
import shutil

def split_data(data_dir, train_dir, test_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
 
    for person_folder in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_folder)
        image_files = os.listdir(person_path)
        print(image_files)
        random.shuffle(image_files)
 
        num_train_images = int(train_ratio * len(image_files))
 
        train_person_dir = os.path.join(train_dir, person_folder)
        test_person_dir = os.path.join(test_dir, person_folder)
 
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)
 
        for i, image_file in enumerate(image_files):
            source_path = os.path.join(person_path, image_file)
            if i < num_train_images:
                destination_path = os.path.join(train_person_dir, image_file)
            else:
                destination_path = os.path.join(test_person_dir, image_file)
 
            if not os.path.isdir(destination_path):
                shutil.copyfile(source_path, destination_path)

def main():
    # Directories
    data_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/data_9thOctManual_croppedUpdated'
    train_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_V14_train'
    test_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_V14_test'

    split_data(data_dir, train_dir, test_dir)
   
if __name__ == "__main__":
    main() 
