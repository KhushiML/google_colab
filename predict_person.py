import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

# Load your trained model
model = load_model('/home/vmadmin/trained_model_vgg_finetuned_today.h5')

# Define the directory containing your training images
train_dir ='/home/vmadmin/cropped_train'

# Create a mapping of class indices to class names
class_labels = os.listdir(train_dir)  # Get the list of class directories
class_labels.sort()  # Sort to ensure consistent indexing
class_indices = {i: label for i, label in enumerate(class_labels)}

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match the input shape
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Predict the class of a new image
def predict_person(img_path):
    processed_image = load_and_preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)  # Get the class with highest probability
    predicted_name = class_indices[predicted_class_index[0]]  # Map index to name
    return predicted_name

# Example usage
if __name__ == '__main__':
    img_path = '/home/vmadmin/Anushka_Sharma.jpg'  # Specify the path to the image you want to predict
    predicted_name = predict_person(img_path)
    print(f'The predicted person is: {predicted_name}')
"""
# Example usage
if __name__ == '__main__':
    # Specify the directory containing the images for prediction
    image_dir = '/home/vmadmin/detected_persons'  # Adjust path accordingly
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):  # Check if the file is a JPEG image
            img_path = os.path.join(image_dir, img_file)  # Full path to the image
            predicted_name = predict_person(img_path)
            print(f'Image: {img_file}, Predicted person: {predicted_name}')
"""
