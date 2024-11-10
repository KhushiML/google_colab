import cv2        
import numpy as np                                                                                                                                  
import tensorflow as tf                                                                                                                             
import requests                                                                                                                                     
from tensorflow.keras.models import load_model                                                                                                     
from tensorflow.keras.preprocessing.image import ImageDataGenerator                                                                                 
import os                                                                                                                                           
from datetime import datetime                                                                                                                       
from PIL import Image                                                                                                                               
from io import BytesIO                                                                                                                             
import base64                                                                                                                                       
import json                                                                                                                                        
import re                                                                                                                                           
import uuid                                                                                                                                         
from google.cloud import storage                                                                                                                  
import time    
import torch
import sys, json
from google.api_core.exceptions import GoogleAPIError

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



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

def save_to_bucket(image, file_name,config):
    bucket_name = config['bucket_name']
    credentials_path = config['credentials_path']
    
    try:
        # Initialize the GCS client and access the bucket
        client = storage.Client.from_service_account_json(credentials_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Convert the image to JPEG format
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image to JPEG format.")

        # Upload the image to the bucket
        blob.upload_from_string(encoded_image.tobytes(), content_type="image/jpeg")
        return blob.public_url

    except GoogleAPIError as e:
        print(f"Google Cloud Storage error: {e}")
    except ValueError as e:
        print(f"Image processing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

def save_unknown_to_db(id, cropped_image_url, video_url, frame_url, config):
    unknown_save_api = config["save_unknown_api"]
    params = {
        "substory_id": id,
        "per_name": "UNKNOWN",
        "cropped_image_url": cropped_image_url,
        "clip_url": video_url,
        "frame": frame_url
    }

    try:
        #print(params)
        response = requests.post(unknown_save_api, json=params)
        #print(response)
        response.raise_for_status()  # Raises an exception for HTTP errors
        #print(response.json())
        return response

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

def update_database(api_endpoint, video_url, frame, personality_name, frame_time):
    params = {
        "video_url": video_url,
        "personality_name": personality_name,
        "frame": frame,
        "frame_time": frame_time
    }

    try:
        #print(params)
        # Make an HTTP POST request to the specified API endpoint
        response = requests.post(api_endpoint, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
        
        return response

    except Exception as e:
        print(f"An error occurred while updating the database: {e}")  # Handle all exceptions

    return None  # Return None if any error occurs

def detect_face(image):
    #image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_eye_data = []
    
    # Loop over detected faces to detect eyes within each face
    for (x, y, w, h) in faces:
        roi_gray = image[y:y+h, x:x+w]  # Region of Interest (face area)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Only consider the face if at least two eyes are detected
        if len(eyes) >= 1:
            face_eye_data.append(((x, y, w, h), eyes))

    return face_eye_data 


def predict_person(api_endpoint, model, video_url, frame_url, frame_time, train_generator, id, config, threshold=0.85):
    try:
        image = process_image(frame_url)
        face_eye_data = detect_face(image)

        if len(face_eye_data) == 0:
            print("No face with two eyes found")
            return []

        print("Face eye data", face_eye_data)

        predictions = []
        person_names = list(train_generator.class_indices.keys())

        for i, ((x, y, w, h), eyes) in enumerate(face_eye_data):
            face_image = cv2.resize(image[y:y+h, x:x+w], (224, 224))
            face_image = face_image.astype("float32") / 255.0
            face_image = np.expand_dims(face_image, axis=0)

            pred = model.predict(face_image)
            person_index = np.argmax(pred)
            accuracy = pred[0][person_index]

            if accuracy < threshold:
                padding_w = int(1 * w)  # Adjust these multipliers based on the padding you want
                padding_h = int(1 * h)
                
                x_new = max(0, x - padding_w)
                y_new = max(0, y - padding_h)
                w_new = min(image.shape[1] - x_new, w + 2 * padding_w)
                h_new = min(image.shape[0] - y_new, h + 2 * padding_h)

                cropped_face_image = image[y_new:y_new+h_new, x_new:x_new+w_new]
                unknown_face_filename = f"unknown_faces/{uuid.uuid4()}.jpg"
                print("UNKNOWN found accuracy < threshold")
                
                # Save cropped image to GCS
                cropped_image_url = save_to_bucket(cropped_face_image, unknown_face_filename,config)
                print(cropped_image_url)

                # Update database with "UNKNOWN" entry
                response = update_database(api_endpoint, video_url, cropped_image_url, "UNKNOWN", frame_time)
                if response and response.status_code == 200:
                    print("Unknown frame processed and saved successfully.")
                else:
                    print(f"Failed to process frame. Error: {response.text if response else 'No response from API'}")
                
                # Save the unknown face to the master table
                response1 = save_unknown_to_db(id, cropped_image_url, video_url, frame_url, config)
                if response1 and response1.status_code == 201:
                    print("Unknown saved successfully to master table.")
                else:
                    print(f"Failed to save to master table. Error: {response1.text if response1 else 'No response from API'}")
                
                predictions.append("UNKNOWN")
            else:
                predicted_name = person_names[person_index]
                predictions.append(predicted_name)

        print("Predictions", predictions)
        return predictions

    except Exception as e:
        print(f"Error predicting personality: {e}")
        return []


def get_video_url(get_api):
    try:
        # Make an HTTP GET request to the specified API endpoint
        response = requests.get(get_api)
        
        # Check if the request was successful (status code 200)
        response.raise_for_status()  # Raises an HTTPError if the status code indicates an error
        
        # Attempt to parse the response as JSON
        data = response.json()
        return data

    except Exception as e:
        print(f"An error occurred: {e}")  # Handle all exceptions

    return None

def get_frames(api_url, id):
    try:
        # Prepare the request URL with the query parameter 'id'
        full_url = f"{api_url}?id={id}"

        # Send a POST request to the API
        response = requests.post(full_url, headers={"accept": "application/json"})

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            # Select frames based on the number of image URLs
            selected_frames = [ {"image_url": item["image_url"], "frame_datetime": item["frame_datetime"]} for item in data
                if "image_url" in item and "frame_datetime" in item]
            if len(selected_frames) > 10:
                interval = 5  # 10 seconds = 5 frames at 1 frame per 2 seconds
                selected_frames = selected_frames[::interval]
            else:
                selected_frames = selected_frames

            return selected_frames
        else:
            print(f"Failed to get data. Status code: {response.status_code}")
            return []

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def save_category(id, name, category, config):
    save_api_endpoint = config["save_category_api"]
    params = {
        "id": id,
        "name": name,
        "category": "Personality"
    }

    try:
        # Make an HTTP POST request to the save API endpoint
        response = requests.post(save_api_endpoint, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the status code indicates an error
        
        return response

    except Exception as e:
        print(f"An error occurred while saving the category: {e}")  # Handle all exceptions

    return None  # Return None if any error occurs


def set_status(set_api, id):
    params = {
        "id": id
    }

    try:
        # Make an HTTP PATCH request to the set status API endpoint
        response = requests.patch(set_api, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the status code indicates an error
        
        # Print and return the JSON response
        #print(response.json())
        return response

    except Exception as e:
        print(f"An error occurred while setting the status: {e}")  # Handle all exceptions

    return None  # Return None if any error occurs

"""

def process_video(model, train_generator, video_url, selected_frames, api_endpoint, id, text_content, channel_code, config):
    # Ensure directory exists
    check_personality_api = config["check_personality_api"]

    for frame in selected_frames:
        frame_url = frame['image_url']
        frame_datetime = frame['frame_datetime']
        datetime_obj = datetime.fromisoformat(frame_datetime)
        frame_time = datetime_obj.time()

        personality_names = predict_person(api_endpoint, model, video_url, frame_url, frame_time, train_generator, id, config)

        # Analyze the frame using the OpenAI API
        openai_name, openai_category = categorize(frame_url, text_content, config)
        print(openai_name)
        print(openai_category)

        name_to_check = None

        # Determine the name to check
        if openai_name:
            name_to_check = openai_name
        elif personality_names:
            name_to_check = personality_names[0]  # Use the first predicted name if OpenAI fails

        # Check if OpenAI category requires validation
        if openai_category in ["Anchor", "Reporter"] and name_to_check != "UNKNOWN":
            # Validate against both categories
            categories_to_check = ["Anchor", "Reporter"]

            # Check the personality against the master table for both categories
            is_valid = False
            for category in categories_to_check:
                check_response = requests.get(check_personality_api, params={
                    "channel_code": channel_code,
                    "per_name": name_to_check,
                    "per_type": category
                })
                check_data = check_response.json()

                if check_data.get("result"):
                    is_valid = True
                    break  # No need to check further if found valid

            if is_valid:
                # Update the database with the validated name
                response = update_database(api_endpoint, video_url, frame_url, name_to_check, frame_time)
                result = save_category(id, name_to_check, openai_category, config)

                if response.status_code == 200:
                    print(f"Frame processed and saved successfully with name: {name_to_check}.")
                else:
                    print(f"Failed to process frame with name. Error: {response.text}")
            else:
                print(f"Personality '{name_to_check}' not found in the master for channel '{channel_code}'. Discarding this frame.")
                continue  # Discard this frame

        elif openai_name:
            # If category is not Anchor or Reporter, update with OpenAI name
            response = update_database(api_endpoint, video_url, frame_url, openai_name, frame_time)
            result = save_category(id, openai_name, openai_category, config)

            if response.status_code == 200:
                print(f"Frame processed and saved successfully with OpenAI name and category.")
            else:
                print(f"Failed to process frame with OpenAI name and category. Error: {response.text}")

        elif personality_names:
            # No OpenAI name; use the model's first predicted name
            model_name = personality_names[0]
            print(f"No OpenAI name. Using model-predicted name: {model_name}")

            # Update the database without checking personality if category is not Anchor or Reporter
            response = update_database(api_endpoint, video_url, frame_url, model_name, frame_time)
            result = save_category(id, model_name, openai_category, config)

            if response.status_code == 200:
                print(f"Frame processed and saved successfully with model-predicted name.")
            else:
                print(f"Failed to process frame with model-predicted name. Error: {response.text}")

    return None
    
"""

def process_video(model, train_generator, video_url, selected_frames, api_endpoint, id, text_content, channel_code, config):
    # Ensure directory exists
    for frame in selected_frames:
        frame_url = frame['image_url']
        frame_datetime = frame['frame_datetime']
        datetime_obj = datetime.fromisoformat(frame_datetime)
        frame_time = datetime_obj.time()

        # Use the model to predict personality names from the frame
        personality_names = predict_person(api_endpoint, model, video_url, frame_url, frame_time, train_generator, id, config)

        # Determine the name to check using only the model predictions
        name_to_check = personality_names[0] if personality_names else None

        # Update the database with the predicted name
        if name_to_check:
            response = update_database(api_endpoint, video_url, frame_url, name_to_check, frame_time)
            result = save_category(id, name_to_check, None, config)  # No category since OpenAI is not used

            if response.status_code == 200:
                print(f"Frame processed and saved successfully with name: {name_to_check}.")
            else:
                print(f"Failed to process frame with name. Error: {response.text}")

    return None


def update_personality_code(id, config):
    update_api = config["update_personality_code_api"]
    response = requests.patch(update_api, json={"id": id})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to update personality code for ID {id}: {response.text}")
        return None


def load_config():
    with open('config.json') as config_file:
        return json.load(config_file)

def main():
    # Load configuration from JSON
    config = load_config()

    last_mp4_url = None
    get_api = config["get_api"]
    frames_api_url = config["frames_api_url"]
    detect_faces_api = config["detect_faces_api"]
    set_api = config["set_api"]
    
    train_dir = config["train_dir"]
    model = load_model(config["model_path"])

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
        batch_size=16,
        class_mode='categorical'
    )
    
    while True:
        try:
            data = get_video_url(get_api)
            print(data)

            if data is None:
                print("No valid data received. Sleeping for 5 seconds.")
                time.sleep(5)
                continue
            
            video_url = data.get('clip_url')
            id = data.get('id')
            text_content = data.get('text_content')
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            channel_code = data.get('channel_code')

            if not video_url:
                print("No valid data received. Sleeping for 5 seconds.")
                set_status(set_api, id)
                print(f"Set status for video ID: {id}")
                time.sleep(5)
                continue  # Avoid proceeding with invalid URL

            # Check if start and end dates are not equal
            if start_time and end_time:
                start_dt = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
                end_dt = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
                
                if start_dt.date() != end_dt.date():
                    print(f"Skipping video ID: {id} because start and end dates are not equal.")
                    set_status(set_api, id)
                    continue  # Skip processing this ID

            if video_url and video_url != last_mp4_url:
                print(f"Processing video ID: {id} with channel_code: {channel_code}")

                selected_frames = get_frames(frames_api_url, id)
                if not selected_frames:
                    print(f"Failed to get frames for video ID: {id}. Skipping this video.")
                    set_status(set_api, id)
                    continue

                print(f"Selected frames: {selected_frames}")
                
                try:
                    process_video(model, train_generator, video_url, selected_frames, detect_faces_api, id, text_content, channel_code, config)
                    print(f"Processed video ID: {id}")
                except Exception as e:
                    print(f"Error processing video ID: {id}. Error: {e}")
                    continue  # Skip to the next video
                
                # Update personality codes after processing the video
                update_result = update_personality_code(id, config)
                if update_result is not None:
                    print(f"Updated personality codes for video ID: {id}")
                else:
                    print(f"Failed to update personality codes for video ID: {id}")

                set_status(set_api, id)
                print(f"Set status for video ID: {id}")
                
                last_mp4_url = video_url

                # Kill the process and restart it
                print("Killing the process and restarting...")
                os.execv(sys.executable, ['python'] + sys.argv)

        except Exception as main_error:
            print(f"Error in main loop: {main_error}")
            time.sleep(5)  # Allow some time before retrying

if __name__ == "__main__":
    main()
