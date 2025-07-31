import os
import requests
from flask import Flask, render_template, flash, send_from_directory
import pandas as pd
from urllib.parse import urlparse

# Import the processing functions
import process_logic

# --- Configuration ---
# ‼️ IMPORTANT: CONFIGURE THESE PATHS ‼️
# 1. Path to your Google Cloud credentials JSON file
KEY_PATH = r"D:\National Id Scan\Key.json"

# 2. Hardcoded path for the Excel file
EXCEL_FILE_PATH = r"D:\National Id Scan\nettinghub users ids.xlsx"

# Folder to store downloaded images temporarily
DOWNLOAD_FOLDER = 'static/downloaded_images'

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'

# --- Helper Function ---
def download_image(url, output_path):
    """Downloads a single image from a URL and saves it."""
    if not isinstance(url, str) or not url.strip() or url.lower() == 'null':
        return False, "Invalid or empty URL"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, headers=headers, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return False, f"URL is not an image (Content-Type: {content_type})"

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, "Success"
    except requests.exceptions.RequestException as e:
        return False, f"Download error: {e}"
    except IOError as e:
        return False, f"File save error: {e}"

# --- Main Application Route ---
@app.route('/')
def process_from_excel_links():
    """
    Reads an Excel file, downloads images from a 'back link' for each row,
    processes them, and displays the comparison.
    """
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    if not os.path.exists(EXCEL_FILE_PATH):
        flash(f"Error: Excel file not found at {EXCEL_FILE_PATH}", 'danger')
        return render_template('index.html', results=None)
    try:
        df_full = pd.read_excel(EXCEL_FILE_PATH)
        df = df_full.head(10)
        
        # Check for all required columns
        required_columns = ['id', 'nationality_id', 'back link']
        if not all(col in df.columns for col in required_columns):
            flash(f"Error: One or more required columns ('id', 'nationality_id', 'back link') not found.", 'danger')
            return render_template('index.html', results=None)
    except Exception as e:
        flash(f"Error reading Excel file: {e}", 'danger')
        return render_template('index.html', results=None)

    results = []
    vision_client = process_logic.setup_vision_client(KEY_PATH)
    if not vision_client:
        flash('Could not initialize Google Vision client. Check API key path.', 'danger')
        return render_template('index.html', results=None)

    for index, row in df.iterrows():
        # ‼️ CHANGE: Get the 'id' from the Excel row
        excel_row_id = str(row['id'])
        excel_nationality_id = str(row['nationality_id'])
        image_url = row['back link']
        
        local_filename = f"row_{index + 2}.jpg"
        local_filepath = os.path.join(DOWNLOAD_FOLDER, local_filename)

        success, message = download_image(image_url, local_filepath)
        
        extracted_id = "N/A"
        is_match = False

        if success:
            print(f"Processing downloaded image for row {index + 2}...")
            original_text = process_logic.detect_text_from_image(vision_client, local_filepath)
            extracted_id = process_logic.extract_national_id(original_text) if original_text else "Extraction Failed"
            
            if extracted_id == excel_nationality_id:
                is_match = True
        else:
            extracted_id = f"Download Failed: {message}"

        results.append({
            'excel_row_id': excel_row_id, # Pass the new ID to the template
            'excel_nationality_id': excel_nationality_id,
            'extracted_id': extracted_id,
            'is_match': is_match,
            'image_path': local_filename if success else None
        })

    flash(f"Successfully processed the first {len(df)} rows from the Excel file.", 'success')
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
