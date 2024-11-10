import torch
import open_clip
import cv2
import os
from sentence_transformers import util
from PIL import Image

# Set up device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    return round(float(cos_scores[0][0]) * 100, 2)

def find_matching_persons(test_folder, train_folder, threshold=70.0):
    results = []
    
    # Iterate through each image in the test folder
    for test_image in os.listdir(test_folder):
        test_image_path = os.path.join(test_folder, test_image)
        
        # Check if it's an image file
        if test_image.lower().endswith(('.png', '.jpg', '.jpeg')):
            best_score = 0.0
            matched_name = "Unknown"

            # Compare with each folder in the training directory
            for person_folder in os.listdir(train_folder):
                person_folder_path = os.path.join(train_folder, person_folder)
                
                # Check if it's a directory
                if os.path.isdir(person_folder_path):
                    for train_image in os.listdir(person_folder_path):
                        train_image_path = os.path.join(person_folder_path, train_image)
                        
                        # Calculate similarity score
                        score = generateScore(test_image_path, train_image_path)

                        # Update best score and matched name if score is above the threshold
                        if score > best_score:
                            best_score = score
                            if best_score > threshold:
                                matched_name = person_folder

            results.append((test_image, matched_name, best_score))

    return results

# Example usage
if __name__ == '__main__':
    test_folder = '/home/vmadmin/detected_persons'  # Specify the path to the folder containing images to test
    train_folder = '/home/vmadmin/cropped_train'  # Path to the training directory with subfolders
    results = find_matching_persons(test_folder, train_folder)
    
    for image_name, person_name, score in results:
        print(f"Image: {image_name}, Matched Person: {person_name}, Similarity Score: {score}")
