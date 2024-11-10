import os
import random
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

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

def main():
    # Directories
    train_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/train_27thSep'
    test_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/test_27thSep'
    model_save_path = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/trained_model_vgg_finetuned_V12.h5'

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
