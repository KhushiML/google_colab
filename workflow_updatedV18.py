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

#### Step1: Using NER Model

from flair.data import Sentence
from flair.models import SequenceTagger

# Load the NER tagger
tagger = SequenceTagger.load("ner")

# Function to extract names using Flair NER
def extract_names_from_text(text_content):
    # Create a Sentence object from the text content
    sentence = Sentence(text_content)
    
    # Use the tagger to predict entities
    tagger.predict(sentence)
    
    # Extract named entities of type 'PER' (person)
    names = [entity.text for entity in sentence.get_spans('ner') if entity.tag == 'PER']
    
    # Return the extracted names
    return names
    
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
    
def predict_similar_images(image_path, top_k=5, first_data_index=1):
    # API endpoint
    api_url = 'http://35.208.84.123:8000/search_similar_images'
    
    # API request parameters
    params = {
        'image_path': image_path,
        'top_k': top_k,
        'first_data_index': first_data_index
    }
    
    # Send GET request to the API
    response = requests.get(api_url, params=params, headers={'accept': 'application/json'})
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Check if the response is a list of names
        if isinstance(data, list) and len(data) > 0:
            # Return the first name from the list (index 0)
            return data[0]
        else:
            print("Unexpected response format: ", data)
            return None
    else:
        print(f"API request failed with status code {response.status_code}")
        return None

def detect_face_and_predict(image, output_folder='./detected_faces'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    similar_images = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over detected faces and extract them
    for idx, (x, y, w, h) in enumerate(faces):
        roi = image[y:y+h, x:x+w]  # Region of interest for face
        face_image_path = os.path.join(output_folder, f'face_{idx}.jpg')
        
        # Save the face as a separate image
        cv2.imwrite(face_image_path, roi)
        
        # Get similar images for the detected face
        similar_image = predict_similar_images(face_image_path)
        if similar_image:
            similar_images.append(similar_image)
    
    return similar_images



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


def process_video(video_url, selected_frames, api_endpoint, id, text_content, channel_code, config):
    for frame in selected_frames:
        frame_url = frame['image_url']
        frame_datetime = frame['frame_datetime']
        datetime_obj = datetime.fromisoformat(frame_datetime)
        frame_time = datetime_obj.time()

        # Extract names from text content using NER
        names_from_text = extract_names_from_text(text_content)
        
        # Iterate over all extracted names and update the database
        for name_to_check in names_from_text:
            response = update_database(api_endpoint, video_url, None, name_to_check, None)
            if response.status_code == 200:
                print(f"Name from text '{name_to_check}' updated successfully.")
            else:
                print(f"Failed to update name from text '{name_to_check}'. Error: {response.text}")

        # Process the frame image URL
        frame_image = process_image(frame_url)

        if frame_image is not None:
            # Detect faces and predict names from the processed frame image
            similar_images = detect_face_and_predict(frame_image)
            
            # Iterate over all similar images and update the database
            for name_to_check in similar_images:
                response = update_database(api_endpoint, video_url, frame_url, name_to_check, frame_time)
                result = save_category(id, name_to_check, None, config)

                if response.status_code == 200:
                    print(f"Frame processed and saved successfully with name: {name_to_check}.")
                else:
                    print(f"Failed to process frame with name '{name_to_check}'. Error: {response.text}")

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
            text_content = data.get('substory')
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
                    process_video(video_url, selected_frames, detect_faces_api, id, text_content, channel_code, config)
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
