import requests
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO

def get_frames(api_url, id):
    try:
        full_url = f"{api_url}?id={id}"
        response = requests.post(full_url, headers={"accept": "application/json"})

        if response.status_code == 200:
            data = response.json()
            selected_frames = [
                {"image_url": item["image_url"], "frame_datetime": item["frame_datetime"]}
                for item in data if "image_url" in item and "frame_datetime" in item
            ]
            if len(selected_frames) > 10:
                selected_frames = selected_frames[::5]  # Select every 5th frame
            return selected_frames
        else:
            print(f"Failed to get data. Status code: {response.status_code}")
            return []

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def process_image(image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        # Open the image with Pillow
        image = Image.open(BytesIO(response.content))

        # Convert the image to RGB (OpenCV uses BGR by default)
        image = image.convert('RGB')

        # Convert Pillow image to NumPy array
        image_np = np.array(image)

        # Convert RGB to BGR (OpenCV uses BGR color format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return image_bgr
    else:
        print(f"Failed to retrieve image from URL: {image_url}")
        return None     

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_eye_data = []
    
    for (x, y, w, h) in faces:
        roi_gray = image[y:y+h, x:x+w]  # Region of Interest (face area)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 1:
            face_eye_data.append(((x, y, w, h), eyes))

    return face_eye_data 

def process_selected_frames(selected_frames, video_id):
    folder_name = str(video_id)
    os.makedirs(folder_name, exist_ok=True)

    for i, frame_info in enumerate(selected_frames):
        image_url = frame_info['image_url']
        
        # Process the image
        image_bgr = process_image(image_url)

        if image_bgr is not None:
            # Detect faces and eyes
            face_eye_data = detect_face(image_bgr)

            # Draw rectangles around faces and eyes
            for (x, y, w, h), eyes in face_eye_data:
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw face rectangle
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image_bgr, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)  # Draw eye rectangles

            # Save the image with detections
            cv2.imwrite(os.path.join(folder_name, f"frame_{i}_detected.png"), image_bgr)
            print(f"Saved detected frame as {folder_name}/frame_{i}_detected.png")
        else:
            print(f"Failed to process image from URL: {image_url}")

if __name__ == "__main__":
    api_url = "https://tabsons-fastapi-g55rbik64q-el.a.run.app/get_frames/"
    video_id = "119968"  # Replace with your actual video ID
    selected_frames = get_frames(api_url, video_id)

    if selected_frames:
        process_selected_frames(selected_frames, video_id)
    else:
        print("No frames were returned.")
