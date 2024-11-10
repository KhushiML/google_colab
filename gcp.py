from google.cloud import storage
import os
from datetime import datetime

def download_images_from_gcp(bucket_name, prefix, download_dir):
    # Path to your Google Cloud credentials JSON file
    credentials_path = "/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/tabsons-f3426d01189d.json"
    client = storage.Client.from_service_account_json(credentials_path)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    # Define the cutoff date (September 16, 2024)
    cutoff_date = datetime(2024, 9, 16).date()

    for blob in blobs:
        print(f"Checking blob: {blob.name} updated on {blob.updated.date()}")
        # Check if the blob was updated on or after the cutoff date
        if blob.updated.date() >= cutoff_date:
            if blob.name.endswith(('.jpg', '.jpeg', '.png')):
                subfolder_path = os.path.dirname(blob.name.replace(prefix, '').lstrip('/'))
                local_dir = os.path.join(download_dir, subfolder_path)

                # Create local directory if it doesn't exist
                os.makedirs(local_dir, exist_ok=True)

                file_name = os.path.join(local_dir, os.path.basename(blob.name))

                # Download the file
                blob.download_to_filename(file_name)
                print(f"Downloaded {file_name}")

def main():
    # Replace with your GCP bucket name and desired prefix
    bucket_name = 'imagestg-bucket'
    prefix = 'Personality_Images'
    download_dir = '/home/vmadmin/DATASET_BACKUP/FINAL_DATASET/download_till_9thOctober'

    download_images_from_gcp(bucket_name, prefix, download_dir)

if __name__ == "__main__":
    main()
