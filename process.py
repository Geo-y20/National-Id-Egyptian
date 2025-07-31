import re
import io
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from google.cloud import vision
from google.oauth2 import service_account

# --- Google Vision API Setup ---
def setup_vision_client(key_path: str):
    """Initializes and returns a Google Vision API client."""
    try:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        print(f"❌ Critical Error: Could not setup Google Vision client. Check your key path. Error: {e}")
        return None

def detect_rotation_and_correct_image(client, image_path: str) -> str:
    """Detects image rotation and corrects it."""
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.text_annotations:
            vertices = response.text_annotations[0].bounding_poly.vertices
            if vertices:
                dy = vertices[1].y - vertices[0].y
                dx = vertices[1].x - vertices[0].x
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if abs(angle) > 45:
                    with PILImage.open(image_path) as img:
                        img = img.convert('RGB')
                        img = img.rotate(270 if angle > 0 else 90, expand=True)
                        corrected_path = image_path.replace('.jpg', '_corrected.jpg')
                        img.save(corrected_path)
                        print(f"🔄 Image rotated and saved as: {corrected_path}")
                        return corrected_path
        return image_path
    except Exception as e:
        print(f"❌ Error correcting image rotation {image_path}: {str(e)}")
        return image_path

def detect_text_from_image(client, image_path: str) -> str | None:
    """Detects and extracts text from an image file after rotation correction."""
    corrected_path = detect_rotation_and_correct_image(client, image_path)
    try:
        with io.open(corrected_path, 'rb') as image_file:
            content = image_file.read()
    except Exception as e:
        print(f"❌ Error reading image file {corrected_path}: {str(e)}")
        return None
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        print(f"❌ Vision API error for {corrected_path}: {response.error.message}")
        return None
    text = response.text_annotations[0].description if response.text_annotations else None
    # Clean up temporary corrected file if it was created
    if corrected_path != image_path and os.path.exists(corrected_path):
        os.remove(corrected_path)
    return text

# --- Information Extraction Logic ---
def normalize_digits(text: str) -> str:
    """
    (IMPROVED) Converts all known variants of Arabic-Indic numerals to Western digits.
    """
    if not text:
        return ""
    # Mapping for both Eastern Arabic and Persian/Urdu numerals
    numeral_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
    }
    translation_table = str.maketrans(numeral_map)
    return text.translate(translation_table)

def extract_national_id(text: str) -> str | None:
    """
    (IMPROVED) Extracts the 14-digit Egyptian National ID using a more robust and safe logic.
    """
    if not text:
        return None

    # Step 1: Normalize all numeral variants to Western digits. This is the most critical fix.
    processed_text = normalize_digits(text)

    # Step 2: Create a "clean" version of the text by removing all date patterns to avoid confusion.
    # This regex removes YYYY/MM/DD, YYYY/MM, and similar formats.
    cleaned_text = re.sub(r'\b\d{4}/\d{1,2}(/\d{1,2})?\b', '', processed_text)

    # Strategy 1: Find a clean, contiguous 14-digit ID in the cleaned text.
    # The ID must start with a '2' or '3'.
    match = re.search(r'\b([23]\d{13})\b', cleaned_text)
    if match:
        print("✅ Strategy 1 Success: Found a contiguous 14-digit ID.")
        return match.group(1)

    # Strategy 2: Find two adjacent 7-digit numbers in the cleaned text.
    all_numbers = re.findall(r'\d+', cleaned_text)
    for i in range(len(all_numbers) - 1):
        if len(all_numbers[i]) == 7 and len(all_numbers[i+1]) == 7:
            national_id = all_numbers[i] + all_numbers[i+1]
            if national_id.startswith(('2', '3')):
                print("✅ Strategy 2 Success: Found two adjacent 7-digit groups.")
                return national_id

    # Strategy 3 (Smarter & Safer Fallback): Look for a 14-digit ID inside longer number sequences.
    # This specifically targets cases where a date was merged with the ID by OCR.
    all_numbers_original = re.findall(r'\d+', processed_text) # Use original text with dates for this check
    for num_seq in all_numbers_original:
        if len(num_seq) > 14:
            # Search for a valid ID embedded within the long number
            embedded_match = re.search(r'([23]\d{13})', num_seq)
            if embedded_match:
                print("⚠️ Strategy 3 (Smarter Fallback) Success: Found embedded ID.")
                return embedded_match.group(1)
    
    # If all else fails, return nothing. This prevents creating bad data.
    return None

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- ‼️ IMPORTANT: CONFIGURE THESE TWO PATHS ‼️ ---
    # 1. Path to your Google Cloud credentials JSON file
    KEY_PATH = r"D:\National Id Scan\Key.json"
    
    # 2. Path to the folder containing your ID images
    IMAGE_FOLDER = r"D:\National Id Scan\national_id_images"
    # ---------------------------------------------------

    print("--- Starting ID Scan Process ---")
    vision_client = setup_vision_client(KEY_PATH)
    
    if vision_client and os.path.isdir(IMAGE_FOLDER):
        # Sort files to process them in a consistent order (e.g., image_1, image_2...)
        files_to_process = sorted(os.listdir(IMAGE_FOLDER))
        for filename in files_to_process:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(IMAGE_FOLDER, filename)
                print(f"\n\n========================================")
                print(f"🔎 Processing Image: {filename}")
                print(f"========================================")
                
                original_text = detect_text_from_image(vision_client, image_path)
                
                if original_text:
                    print("\n--- Extracted Original Text ---")
                    print(original_text)
                    print("-------------------------------\n")

                    national_id_found = extract_national_id(original_text)
                    
                    if national_id_found:
                        print(f"🎉 SUCCESS: Detected National ID is {national_id_found}")
                    else:
                        print(f"❌ FAILED: Could not detect a National ID in {filename}.")
                else:
                    print(f"Could not extract any text from {filename}.")
    elif not vision_client:
        print("Could not start. Please check the Google Vision API key path.")
    else:
        print(f"Error: The folder '{IMAGE_FOLDER}' does not exist.")