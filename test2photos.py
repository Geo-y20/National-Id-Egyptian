import re
import os
import io
from google.cloud import vision
from google.oauth2 import service_account

# --- Google Vision API Setup ---
def setup_vision_client(key_path: str):
    """Initializes and returns a Google Vision API client."""
    credentials = service_account.Credentials.from_service_account_file(key_path)
    return vision.ImageAnnotatorClient(credentials=credentials)

def detect_text_from_image(client, image_path: str) -> str | None:
    """Detects and extracts text from an image file."""
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
    except Exception as e:
        print(f"❌ Error reading image file {image_path}: {str(e)}")
        return None
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    
    if response.error.message:
        print(f"❌ Vision API error for {image_path}: {response.error.message}")
        return None
    
    # Return the original, unmodified text for processing
    return response.full_text_annotation.text

# --- Information Extraction Logic ---
def extract_national_id(text: str) -> str | None:
    """
    Extracts the 14-digit Egyptian National ID by searching line-by-line
    and using a comprehensive numeral conversion map.

    Args:
        text: The raw text extracted from the ID card.

    Returns:
        The 14-digit National ID as a string, or None if not found.
    """
    if not text:
        return None
    
    # This map now includes both standard AND Eastern Arabic-Indic numerals.
    arabic_to_western_map = str.maketrans(
        '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', 
        '01234567890123456789'
    )
    
    # Search for the ID line by line to avoid mixing numbers from different fields.
    for line in text.split('\n'):
        # Normalize the digits in the current line
        normalized_line = line.translate(arabic_to_western_map)
        
        # Find all digits in the line and join them
        digits_in_line = "".join(re.findall(r'\d+', normalized_line))
        
        # Check if this line contains the 14-digit ID
        if len(digits_in_line) == 14 and digits_in_line.startswith(('2', '3')):
            return digits_in_line
            
    return None

def extract_full_name(text: str) -> str | None:
    """
    Extracts the full name, which is located after the "بطاقة تحقيق الشخصية" line.

    Args:
        text: The raw text extracted from the ID card.

    Returns:
        The extracted full name, or None if not found.
    """
    if not text:
        return None
    
    lines = text.split('\n')
    
    try:
        header_index = next(i for i, line in enumerate(lines) if "بطاقة تحقيق الشخصية" in line)
        for i in range(1, 3):
            if header_index + i < len(lines):
                name_candidate = lines[header_index + i].strip()
                is_valid_name = (
                    2 <= len(name_candidate.split()) <= 4 and
                    bool(re.fullmatch(r'[\u0600-\u06FF\s]+', name_candidate)) and
                    not any(char.isdigit() for char in name_candidate)
                )
                if is_valid_name:
                    return name_candidate
    except StopIteration:
        return None
    return None

# --- Main Processing Flow ---
def main():
    """
    Main function to set up the client and process a directory of images.
    """
    print("🚀 Starting Egyptian ID Processing Script...")

    # --- Configuration ---
    key_path = r"D:\National Id Scan\Key.json" 
    image_directory = r"D:\National Id Scan\static\national_id_images"

    if not os.path.exists(key_path) or not os.path.isdir(image_directory):
        print("❌ CRITICAL ERROR: Check your key_path and image_directory.")
        return

    client = setup_vision_client(key_path)
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("🤷 No image files found in the specified directory.")
        return

    for image_path in image_files:
        print(f"\n📄 Processing Image: {os.path.basename(image_path)}")
        original_text = detect_text_from_image(client, image_path)
        
        if original_text:
            print("\n--- Original Extracted Text ---")
            print(original_text)
            print("-----------------------------\n")

            full_name = extract_full_name(original_text)
            national_id = extract_national_id(original_text)
            
            print(f"  ✅ Extracted Name: {full_name or 'Not Found'}")
            print(f"  ✅ Extracted National ID: {national_id or 'Not Found'}")
        else:
            print("  - Could not extract text from this image.")
        print("-" * 50)

if __name__ == "__main__":
    main()