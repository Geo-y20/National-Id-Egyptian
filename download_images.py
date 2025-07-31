import pandas as pd
import requests
import os
from urllib.parse import urlparse

def download_images_from_excel(excel_path, output_folder='downloaded_images'):
    """
    Reads an Excel file, downloads images from a 'back link' column, and saves them to a specified folder.

    Args:
        excel_path (str): The full path to the Excel file.
        output_folder (str): The name of the folder to save images in.
    """
    try:
        # Read the entire Excel file into a pandas DataFrame
        df = pd.read_excel(excel_path)

        # Print the first 50 rows of the DataFrame for preview
        print("--- First 50 rows of your Excel file ---")
        print(df.head(50))
        print("-----------------------------------------")

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created folder: {output_folder}")

        # Check if the 'back link' column exists
        if 'back link' not in df.columns:
            print("Error: 'back link' column not found in the Excel file.")
            return

        # Loop through each row in the DataFrame to download images
        for index, row in df.iterrows():
            image_url = row['back link']
            if not isinstance(image_url, str) or not image_url.strip() or image_url.lower() == 'null':
                print(f"Skipping row {index + 1}: 'back link' is empty, invalid, or NULL.")
                continue

            try:
                # Get the image content with a timeout
                headers = {'User-Agent': 'Mozilla/5.0'}  # Add user-agent to avoid server blocks
                response = requests.get(image_url, stream=True, headers=headers, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Check if the content type is an image
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    print(f"Skipping row {index + 1}: URL {image_url} is not an image (Content-Type: {content_type}).")
                    continue

                # Extract file extension from URL or default to .jpg
                parsed_url = urlparse(image_url)
                file_ext = os.path.splitext(parsed_url.path)[1].lower()
                if not file_ext in ['.jpg', '.jpeg', '.png']:
                    file_ext = '.jpg'  # Default to .jpg if extension is unknown

                # Create a filename for the image
                filename = os.path.join(output_folder, f"image_{index + 1}{file_ext}")

                # Save the image to the output folder
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Successfully downloaded and saved {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {image_url}: {e}")
            except IOError as e:
                print(f"Error saving file {filename}: {e}")

    except FileNotFoundError:
        print(f"Error: The file at {excel_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Path to your Excel file
    excel_file_path = r'D:\National Id Scan\nettinghub users ids.xlsx'

    # Output folder name
    output_directory = 'national_id_images'

    download_images_from_excel(excel_file_path, output_directory)