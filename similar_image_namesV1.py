import requests

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


image_path = '/home/vmadmin/testing_images/siddiqui1.jpg'

# Get the first similar image name (default values)
first_similar_image_name = predict_similar_images(image_path)

if first_similar_image_name:
    print(f"The first similar image name is: {first_similar_image_name}")
else:
    print("No similar images found.")

