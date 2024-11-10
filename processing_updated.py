import cv2
import os

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_faces(input_dir, output_dir):
    # Iterate over all folders in the input directory
    for person_folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        # Create output directory for cropped faces
        person_output_dir = os.path.join(output_dir, person_folder)
        os.makedirs(person_output_dir, exist_ok=True)
        
        # Iterate over all images in the person folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            # Skip non-image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Read the image
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Unable to read image {image_path}. Skipping...")
                continue
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"No face detected in {image_path}.")
                continue
            
            face_detected = False
            
            # Iterate over detected faces
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]

                # Ensure the cropped face image is valid
                if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                    print(f"Cropped face image is invalid for {image_path}. Skipping...")
                    continue
                
                # Save the face image
                output_file = os.path.join(person_output_dir, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
                cv2.imwrite(output_file, face_img)
                print(f"Face cropped and saved to {output_file}")
                face_detected = True

            if not face_detected:
                print(f"No valid faces detected in {image_path}.")

# Paths to the dataset
train_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/V14_test184'
train_output_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/V14_test184_cropped'

# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)

# Process training dataset
detect_and_crop_faces(train_dir, train_output_dir)
