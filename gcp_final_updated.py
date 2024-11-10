from google.cloud import storage
import os
from datetime import datetime, timezone

def download_images_from_gcp(bucket_name, prefix, download_dir, start_date, end_date):
    # Path to your Google Cloud credentials JSON file
    credentials_path = "/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/tabsons-f3426d01189d.json"
    client = storage.Client.from_service_account_json(credentials_path)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # Print the name of the blob and its update date
        print(f"Checking blob: {blob.name} updated on {blob.updated}")

        # Check if the blob was updated within the specified date range
        if start_date <= blob.updated <= end_date:
            # Create subfolder path
            subfolder_path = os.path.dirname(blob.name.replace(prefix, '').lstrip('/'))
            local_dir = os.path.join(download_dir, subfolder_path)

            # Create local directory if it doesn't exist
            os.makedirs(local_dir, exist_ok=True)

            # Create a file name with .jpg extension
            file_name = os.path.join(local_dir, os.path.basename(blob.name))

            # Rename to .jpg if not already an image type
            if not blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_name = f"{file_name.rsplit('.', 1)[0]}.jpg"  # Replace any existing extension with .jpg

            # Download the file
            blob.download_to_filename(file_name)
            print(f"Downloaded {file_name}")

def main():
    # Replace with your GCP bucket name and desired prefix
    bucket_name = 'imagestg-bucket'
    prefix = 'Personality_Images'
    download_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/download_check'
    
    # Define the date range for downloading images
    start_date = datetime(2024, 9, 16, tzinfo=timezone.utc)  # Make it timezone-aware
    end_date = datetime.now(timezone.utc)                     # Set end date to today's date, timezone-aware

    download_images_from_gcp(bucket_name, prefix, download_dir, start_date, end_date)

if __name__ == "__main__":
    main()
