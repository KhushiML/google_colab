def process_video(model, train_generator, video_url, selected_frames, api_endpoint, id, text_content):
    # Extract channel_code from the substory table
    channel_code = get_channel_code_from_id(id)  # Implement this function to fetch channel_code based on the substory ID

    for frame in selected_frames:
        frame_url = frame['image_url']
        frame_datetime = frame['frame_datetime']
        datetime_obj = datetime.fromisoformat(frame_datetime)
        frame_time = datetime_obj.time()

        personality_names = predict_person(api_endpoint, model, video_url, frame_url, frame_time, train_generator, id)

        # Analyze the frame using the OpenAI API
        openai_name, openai_category = categorize(frame_url, text_content)
        print(f"OpenAI Name: {openai_name}, Category: {openai_category}")

        # If no OpenAI name is found, fallback to the first predicted name or "UNKNOWN"
        if not openai_name:
            openai_name = personality_names[0] if personality_names else "UNKNOWN"
            print(f"No OpenAI name found. Using predicted name: {openai_name}")

        # Save the predicted name and category to the database
        response = update_database(api_endpoint, video_url, frame_url, openai_name, frame_time)
        save_category(id, openai_name, openai_category)

        if response.status_code == 200:
            print("Frame processed and saved successfully.")
        else:
            print(f"Failed to process frame. Error: {response.text}")

        # Check if the category is "anchor" or "reporter"
        if openai_category.lower() in ['anchor', 'reporter']:
            personality_check_api = "https://example.com/personality-master-check"  # Replace with actual API URL
            personality_check_response = requests.get(personality_check_api, params={"category": openai_category, "channel_code": channel_code})

            if personality_check_response.status_code == 200:
                personality_data = personality_check_response.json()
                if personality_data.get('exists'):
                    print(f"Personality found in master for category: {openai_category} and channel code: {channel_code}.")
                    # Update the database if personality is found in master
                    update_response = update_database(api_endpoint, video_url, frame_url, openai_name, frame_time)
                    if update_response.status_code == 200:
                        print("Personality updated successfully in the database.")
                    else:
                        print(f"Failed to update personality in the database. Error: {update_response.text}")
                else:
                    print(f"No matching personality found in master for category: {openai_category} and channel code: {channel_code}.")
            else:
                print(f"Failed to check personality master API. Error: {personality_check_response.text}")

    return None
