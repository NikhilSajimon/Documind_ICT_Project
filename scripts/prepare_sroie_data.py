import os
import json
import pickle
import csv
from PIL import Image
from transformers import AutoTokenizer
import numpy as np

# --- CONFIGURATION (Assumed Path Structure) ---
SROIE_BOX_DIR = 'data/raw/sroie/ICDAR-2019-SROIE/data/box/'
SROIE_KEY_DIR = 'data/raw/sroie/ICDAR-2019-SROIE/data/key/'
SROIE_IMG_DIR = 'data/raw/sroie/ICDAR-2019-SROIE/data/img/'
PROCESSED_FILE_SROIE = 'data/processed/sroie_aligned_data.pkl'

# --- Universal Label Mapping ---
LABEL_MAP = {
    "O": 0, "B-TOTAL_AMOUNT": 1, "I-TOTAL_AMOUNT": 2,
    "B-INVOICE_DATE": 3, "I-INVOICE_DATE": 4, "B-VENDOR_NAME": 5, "I-VENDOR_NAME": 6,
    "B-VENDOR_ADDRESS": 7, "I-VENDOR_ADDRESS": 8, "B-FIELD_KEY": 9, "I-FIELD_KEY": 10,
    "B-FIELD_VALUE": 11, "I-FIELD_VALUE": 12, "B-DOCUMENT_ID": 13, "I-DOCUMENT_ID": 14,
    "B-LINE_ITEM": 15, "I-LINE_ITEM": 16,
}
SROIE_KEY_MAP = {
    'company': 'VENDOR_NAME', 'date': 'INVOICE_DATE',
    'address': 'VENDOR_ADDRESS', 'total': 'TOTAL_AMOUNT'
}

# --- Utility Functions (Provided in full) ---
def normalize_single_coord(coord_value, original_dimension):
    if original_dimension == 0: return 0
    normalized = round((coord_value / original_dimension) * 1000)
    return max(0, min(1000, normalized))

def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    x0_norm = normalize_single_coord(x0, width)
    x1_norm = normalize_single_coord(x1, width)
    y0_norm = normalize_single_coord(y0, height)
    y1_norm = normalize_single_coord(y1, height)
    return [x0_norm, y0_norm, x1_norm, y1_norm]

def get_image_dimensions(doc_id, image_dir=SROIE_IMG_DIR):
    image_path = os.path.join(image_dir, f'{doc_id}.jpg')
    try:
        with Image.open(image_path) as img:
            return img.size
    except FileNotFoundError:
        return 0, 0

def get_sroie_document_ids(directory):
    return [f.replace('.json', '') for f in os.listdir(directory) if f.endswith('.json')]

# --- SROIE SPECIFIC HELPERS ---

def get_sroie_key_data(doc_id):
    key_path = os.path.join(SROIE_KEY_DIR, f'{doc_id}.json')
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def parse_sroie_box_file(doc_id):
    box_path = os.path.join(SROIE_BOX_DIR, f'{doc_id}.csv')
    if not os.path.exists(box_path):
        return None
    
    tokens, raw_bboxes = [], []
    
    try:
        with open(box_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) < 8: continue
                try:
                    raw_box = [int(p.strip()) for p in row[:4]]
                    token = row[-1].strip()
                    if token:
                        tokens.append(token)
                        raw_bboxes.append(raw_box)
                except ValueError:
                    continue
    except Exception:
        return None
    
    return tokens, raw_bboxes


def align_and_tokenize_sroie(doc_id, tokenizer, label_map, sroie_key_map):
    """Core function to process one SROIE document, passing numerical IDs directly."""
    
    parsed_box_data = parse_sroie_box_file(doc_id)
    key_data = get_sroie_key_data(doc_id)
    width, height = get_image_dimensions(doc_id, SROIE_IMG_DIR)

    # CRITICAL CHECK: Ensure all components are present.
    if not parsed_box_data or not key_data or width == 0 or height == 0:
        # print(f"DEBUG: Skipping {doc_id} due to missing files or zero dimensions.") # Debug print
        return None
        
    tokens, raw_bboxes = parsed_box_data
    word_labels_num = [label_map['O']] * len(tokens)
    
    # --- 1. Apply B-I-O Tags using Key-Value Matching (Numerical Mapping) ---
    full_text_list = [token.lower() for token in tokens]
    full_text_str = " ".join(full_text_list) 
    
    # Flag to track if any key was successfully found
    keys_found_count = 0
    
    for sroie_key, universal_field in sroie_key_map.items():
        if sroie_key in key_data:
            gt_value = key_data[sroie_key].strip().lower() 
            search_start_idx = full_text_str.find(gt_value)
            
            if search_start_idx != -1:
                keys_found_count += 1
                char_count = 0
                match_start_token_idx = -1
                
                # Find the token index where the match starts
                for i, token in enumerate(full_text_list):
                    if char_count <= search_start_idx < char_count + len(token):
                        match_start_token_idx = i
                        break
                    char_count += len(token) + 1 

                # Apply B-I-O tags from the start index
                if match_start_token_idx != -1:
                    is_beginning = True
                    # Set a target length to stop tagging only the matched value (not the rest of the line)
                    gt_value_length = len(gt_value)
                    
                    # Track how much of the ground truth string we have covered
                    coverage_length = 0
                    
                    for i in range(match_start_token_idx, len(tokens)):
                        
                        # Stop tagging if we have covered the entire ground truth length
                        if coverage_length >= gt_value_length:
                            break 
                            
                        tag_str = f'B-{universal_field}' if is_beginning else f'I-{universal_field}'
                        word_labels_num[i] = label_map.get(tag_str, label_map['O'])
                        
                        # Update coverage: Add the token length + 1 (for the space/separator)
                        coverage_length += len(tokens[i]) + 1 
                        
                        is_beginning = False

    # --- CRITICAL FIX: Ensure document has at least one tag before tokenizing ---
    # We only tokenize if we successfully found and tagged at least one key-value pair.
    if keys_found_count == 0:
        # print(f"DEBUG: Skipping {doc_id} because no GT keys were found in OCR text.") # Debug print
        return None
    
    # --- 3. Normalize and Tokenize ---
    normalized_bboxes = [normalize_bbox(box, width, height) for box in raw_bboxes]
    
    # Initialize tokenizer (assuming it's available)
    # NOTE: TOKENIZER MUST BE INITIALIZED ONCE GLOBALLY TO AVOID DOWNLOAD DELAYS
    # For this function, we assume the tokenizer argument is the initialized object.
    
    tokenized_inputs = tokenizer(
        tokens, 
        boxes=normalized_bboxes, 
        word_labels=word_labels_num, 
        truncation=True,
        padding="max_length", 
        return_tensors="pt"
    )

    # The labels tensor now contains the aligned numerical IDs (0-16) and -100 masking
    final_labels = tokenized_inputs.pop("labels").squeeze().tolist()

    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze().tolist(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze().tolist(),
        "bbox": tokenized_inputs["bbox"].squeeze().tolist(),
        "labels": final_labels,
    }


# --- MAIN EXECUTION BLOCK FOR SROIE ---

def process_and_save_sroie(tokenizer, label_map):
    
    sroie_ids = get_sroie_document_ids(SROIE_KEY_DIR) 
    all_processed_data = []
    
    print(f"\nStarting SROIE alignment for {len(sroie_ids)} documents...")
    
    for doc_id in sroie_ids:
        aligned_output = align_and_tokenize_sroie(doc_id, tokenizer, label_map, SROIE_KEY_MAP)
        if aligned_output:
            all_processed_data.append(aligned_output)
            
    # Save the final list of dictionaries to disk
    os.makedirs(os.path.dirname(PROCESSED_FILE_SROIE), exist_ok=True)
    with open(PROCESSED_FILE_SROIE, 'wb') as f:
        pickle.dump(all_processed_data, f)
        
    print(f"✅ Successfully processed SROIE: {len(all_processed_data)} documents.")
    print(f"Data saved to: {PROCESSED_FILE_SROIE}")


if __name__ == '__main__':
    # Initialize the tokenizer component directly
    TOKENIZER = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
    
    # Run the processing and saving routine
    process_and_save_sroie(TOKENIZER, LABEL_MAP)