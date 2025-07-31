import re
import io
import os
import cv2
import math
from PIL import Image as PILImage
from google.cloud import vision
from google.oauth2 import service_account

# This file contains the core logic for image processing and text extraction.
# It's imported by the main app.py file.

def setup_vision_client(key_path: str):
    """Initializes and returns a Google Vision API client."""
    try:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        print(f"❌ Critical Error: Could not setup Google Vision client. Check key path. Error: {e}")
        return None

def detect_rotation_and_correct_image(client, image_path: str) -> str:
    """Detects image rotation and corrects it."""
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        # ‼️ FIX: Check if the image content is empty before making an API call.
        if not content:
            print(f"⚠️ Warning: Content for rotation check is empty for {image_path}. Skipping rotation.")
            return image_path

        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.text_annotations:
            vertices = response.text_annotations[0].bounding_poly.vertices
            if vertices:
                dy = vertices[1].y - vertices[0].y
                dx = vertices[1].x - vertices[0].x
                angle = (180 / 3.14159) * math.atan2(dy, dx)
                if abs(angle) > 45:
                    with PILImage.open(image_path) as img:
                        img = img.convert('RGB')
                        img = img.rotate(270 if angle > 0 else 90, expand=True)
                        corrected_path = image_path.replace('.jpg', '_corrected.jpg').replace('.png', '_corrected.png')
                        img.save(corrected_path)
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

    # ‼️ FIX: Check if the image content is empty before making the main API call.
    if not content:
        print(f"❌ Error: Image content is empty for {corrected_path}. Skipping main text detection.")
        return None

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        print(f"❌ Vision API error for {corrected_path}: {response.error.message}")
        return None
    text = response.text_annotations[0].description if response.text_annotations else None
    if corrected_path != image_path and os.path.exists(corrected_path):
        os.remove(corrected_path)
    return text

def normalize_digits(text: str) -> str:
    """Converts all known variants of Arabic-Indic numerals to Western digits."""
    if not text:
        return ""
    numeral_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
    }
    translation_table = str.maketrans(numeral_map)
    return text.translate(translation_table)

def extract_national_id(text: str) -> str | None:
    """Extracts the 14-digit Egyptian National ID using a robust, multi-step strategy."""
    if not text:
        return None
    processed_text = normalize_digits(text)
    cleaned_text = re.sub(r'\b\d{4}/\d{1,2}(/\d{1,2})?\b', '', processed_text)
    
    match = re.search(r'\b([23]\d{13})\b', cleaned_text)
    if match:
        return match.group(1)

    all_numbers = re.findall(r'\d+', cleaned_text)
    for i in range(len(all_numbers) - 1):
        if len(all_numbers[i]) == 7 and len(all_numbers[i+1]) == 7:
            national_id = all_numbers[i] + all_numbers[i+1]
            if national_id.startswith(('2', '3')):
                return national_id

    all_numbers_original = re.findall(r'\d+', processed_text)
    for num_seq in all_numbers_original:
        if len(num_seq) > 14:
            embedded_match = re.search(r'([23]\d{13})', num_seq)
            if embedded_match:
                return embedded_match.group(1)
    
    return None
