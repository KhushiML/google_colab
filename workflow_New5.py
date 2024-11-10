import os
import random
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import torch
from PIL import Image
from google.cloud import storage
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

def download_images_from_gcp(bucket_name, prefix, download_dir, start_date, end_date):
    # Path to your Google Cloud credentials JSON file
    credentials_path = "/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/tabsons-f3426d01189d.json"
    client = storage.Client.from_service_account_json(credentials_path)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # Print the name of the blob and its update date
        print(f"Checking blob: {blob.name} updated on {blob.updated.date()}")
        
        # Check if the blob was updated within the specified date range
        if start_date <= blob.updated.date() <= end_date:
            if blob.name.endswith(('.jpg', '.jpeg', '.png')):
                subfolder_path = os.path.dirname(blob.name.replace(prefix, '').lstrip('/'))
                local_dir = os.path.join(download_dir, subfolder_path)

                # Create local directory if it doesn't exist
                os.makedirs(local_dir, exist_ok=True)

                file_name = os.path.join(local_dir, os.path.basename(blob.name))

                # Download the file
                blob.download_to_filename(file_name)
                print(f"Downloaded {file_name}")

# 2. Detect and remove corrupt images
def check_and_list_corrupt_images(directory):
    corrupt_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
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

# 3. Detect and crop faces
def detect_and_crop_faces(input_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for person_folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
        person_output_dir = os.path.join(output_dir, person_folder)
        os.makedirs(person_output_dir, exist_ok=True)
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = cv2.imread(image_path)
            if img is None:
                print(f"Unable to read image {image_path}. Skipping...")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                output_file = os.path.join(person_output_dir, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
                cv2.imwrite(output_file, face_img)
                print(f"Face cropped and saved to {output_file}")
                
def remove_and_trim_subfolders(base_dir, threshold=40):
    """Remove subfolders with file counts less than or equal to the specified threshold
       and trim subfolders with more than the threshold to only contain the threshold number of files."""
    
    # Function to count files in a directory
    def count_files(directory):
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    # Function to keep only 'n' files in a directory and remove the rest
    def trim_to_n_files(directory, n):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if len(files) > n:
            files.sort()  # Sort files for consistent ordering (optional)
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
            
            # Check if the file count is less than or equal to the threshold
            if file_count < threshold:
                # Print the name of the subfolder that will be removed
                print(f"Removing {subfolder_path} with {file_count} files")
                
                # Remove the subfolder
                shutil.rmtree(subfolder_path)
            elif file_count > threshold:
                # Trim the subfolder to only keep the first 'threshold' files
                print(f"Trimming {subfolder_path} from {file_count} files to {threshold} files")
                trim_to_n_files(subfolder_path, threshold)
            else:
                print(f"{subfolder_path} has exactly {threshold} files, keeping it.")


# 4. Split the data into training and test sets
def split_data(data_dir, train_dir, test_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for person_folder in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_folder)
        image_files = os.listdir(person_path)
        random.shuffle(image_files)
        num_train_images = int(train_ratio * len(image_files))
        train_person_dir = os.path.join(train_dir, person_folder)
        test_person_dir = os.path.join(test_dir, person_folder)
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)
        for i, image_file in enumerate(image_files):
            source_path = os.path.join(person_path, image_file)
            destination_path = os.path.join(train_person_dir if i < num_train_images else test_person_dir, image_file)
            if not os.path.isdir(destination_path):
                shutil.copyfile(source_path, destination_path)
                
# Function to delete matching folders in additional data directories
def delete_matching_folders(src_dir, target_dir):
    for folder in os.listdir(src_dir):
        src_folder_path = os.path.join(src_dir, folder)
        target_folder_path = os.path.join(target_dir, folder)
        
        # Check if the folder exists in the target directory
        if os.path.isdir(target_folder_path):
            # If it exists, delete the folder in the source directory
            shutil.rmtree(src_folder_path)

def build_vgg16_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_data_generators(train_dir, test_dir, batch_size=16):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

def train_model(model, train_generator, test_generator, epochs=40):
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator)
    )

    return history

def save_model(model, file_path):
    model.save(file_path)

# Main function
def main():
    bucket_name = 'imagestg-bucket'
    prefix = 'Personality_Images'
    download_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/download_check'
    
    # Define the date range for downloading images
    start_date = datetime(2024, 9, 16).date()  # Adjust as needed
    end_date = datetime.now().date()           # Set end date to today's date

    download_images_from_gcp(bucket_name, prefix, download_dir, start_date, end_date)

    # Detect and remove corrupt images
    #train_dir = "/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/download_till_today"
    corrupt_files = check_and_list_corrupt_images(download_dir)
    remove_files(corrupt_files)

    # Detect and crop faces
    detect_and_crop_faces(download_dir, '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_till_today')
    
    base_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_till_today'
    remove_and_trim_subfolders(base_dir)

    # Split data
    split_data(base_dir, '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_train',
               '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_test')
    
    train_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_train'
    test_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_test'
    
    # Define paths for additional data
    additional_train_data = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_train_previous'  # Adjust this path
    additional_test_data = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/cropped_test_previous'    # Adjust this path
    
    # Delete matching folders in additional train data
    delete_matching_folders(additional_train_data, train_dir)

    # Copy additional train data to train_dir, merging subfolders
    shutil.copytree(additional_train_data, train_dir, dirs_exist_ok=True)

    # Delete matching folders in additional test data
    delete_matching_folders(additional_test_data, test_dir)

    # Copy additional test data to test_dir, merging subfolders
    shutil.copytree(additional_test_data, test_dir, dirs_exist_ok=True)
    
    model_save_path = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/trained_model_vgg_finetuned_today.h5'

    # Create data generators
    train_generator, test_generator = create_data_generators(train_dir, test_dir)

    # Determine the number of classes from the training data
    num_classes = len(train_generator.class_indices)
    print(num_classes)

    # Build the model
    model = build_vgg16_model(num_classes=num_classes)

    # Train the model
    history = train_model(model, train_generator, test_generator)

    save_model(model, model_save_path)

if __name__ == "__main__":
    main()
