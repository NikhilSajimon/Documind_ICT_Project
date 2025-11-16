import pickle
import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from collections import defaultdict

# --- CONFIGURATION AND UTILITIES ---

# --- Configuration (UPDATE PATHS AS NECESSARY) ---
JSON_DIR = 'data/raw/funsd/dataset/training_data/annotations'
IMAGE_DIR = 'data/raw/funsd/dataset/training_data/images' 
PROCESSED_FILE = 'data/processed/funsd_train_aligned_data.pkl'

# --- Universal Label Mapping (17 classes + O=0) ---
LABEL_MAP = {
    "O": 0, "B-TOTAL_AMOUNT": 1, "I-TOTAL_AMOUNT": 2,
    "B-INVOICE_DATE": 3, "I-INVOICE_DATE": 4, "B-VENDOR_NAME": 5, "I-VENDOR_NAME": 6,
    "B-VENDOR_ADDRESS": 7, "I-VENDOR_ADDRESS": 8, "B-FIELD_KEY": 9, "I-FIELD_KEY": 10,
    "B-FIELD_VALUE": 11, "I-FIELD_VALUE": 12, "B-DOCUMENT_ID": 13, "I-DOCUMENT_ID": 14,
    "B-LINE_ITEM": 15, "I-LINE_ITEM": 16,
}
# IMPORTANT: The tokenizer uses -100 for padding/special tokens.

# --- Helper Functions (Assuming they are defined in their respective files) ---
# NOTE: If running as a single file, the definitions for normalize_bbox and get_all_document_ids must be present.
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

def get_image_dimensions(doc_id, image_dir=IMAGE_DIR):
    image_path = os.path.join(image_dir, f'{doc_id}.png')
    try:
        with Image.open(image_path) as img:
            return img.size 
    except FileNotFoundError:
        return 0, 0

def get_all_document_ids(json_dir):
    return [f.replace('.json', '') for f in os.listdir(json_dir) if f.endswith('.json')]
# ----------------------------------------------------------------------------------


def align_and_tokenize(doc_id, annotation_data, width, height, tokenizer, label_map):
    """
    FIXED: Converts string labels to numerical IDs (word_labels_num) before tokenization.
    """
    words = []
    word_boxes = []
    word_labels_num = [] # Stores numerical IDs (integers)
    
    # --- Step 1: Extract and Tag at Word Level ---
    for entity in annotation_data.get('form', []):
        fun_label = entity.get('label')
        
        # Determine the Universal Field Tag (string)
        if fun_label == 'question':
            base_label = 'FIELD_KEY'
        elif fun_label == 'answer':
            base_label = 'FIELD_VALUE'
        else: 
            base_label = 'O'
            
        for i, word_obj in enumerate(entity.get('words', [])):
            word_text = word_obj.get('text')
            raw_box = word_obj.get('box')
            
            if not word_text or len(raw_box) != 4: continue

            # Apply B-I-O logic (string tag)
            bio_tag = 'O'
            if base_label != 'O':
                bio_tag = f'B-{base_label}' if i == 0 else f'I-{base_label}'

            # --- CRITICAL FIX: Convert string tag to numerical ID ---
            numerical_label = label_map.get(bio_tag, -100)
            
            # Normalize box and append data
            normalized_box = normalize_bbox(raw_box, width, height)
            
            words.append(word_text)
            word_boxes.append(normalized_box)
            word_labels_num.append(numerical_label) # APPEND THE NUMBER

    # --- Step 2: Tokenize, Align, and Pad (THE FIXED CALL) ---
    # We call the tokenizer component directly. Word labels are now numerical integers.
    tokenized_inputs = tokenizer(
        words, 
        boxes=word_boxes, 
        word_labels=word_labels_num, # <<< PASS NUMERICAL IDs (integers) HERE
        truncation=True,
        padding="max_length", 
        return_tensors="pt"
    )
    
    # The tokenizer handles subword splitting and padding, returning the aligned numerical IDs.
    aligned_labels_ids = tokenized_inputs.pop("labels").squeeze().tolist()

    # Return as a list of Python types (no further mapping needed for labels)
    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze().tolist(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze().tolist(),
        "bbox": tokenized_inputs["bbox"].squeeze().tolist(),
        "labels": aligned_labels_ids, # Return the list directly
    }


def process_and_save_data(json_dir, image_dir, processed_file, tokenizer, label_map):
    """
    Main loop to align, tokenize, and save the entire dataset.
    """
    full_doc_ids = get_all_document_ids(json_dir)
    all_processed_data = []
    
    print(f"Starting alignment for {len(full_doc_ids)} documents...")
    
    for i, doc_id in enumerate(full_doc_ids):
        # 1. Load Data (dimensions, json)
        width, height = get_image_dimensions(doc_id, image_dir)
        json_path = os.path.join(json_dir, f'{doc_id}.json')

        if width == 0 or height == 0 or not os.path.exists(json_path):
            continue 

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
        except Exception:
            continue
            
        # 2. Perform Alignment and Tokenization
        aligned_output = align_and_tokenize(
            doc_id, 
            annotation_data, 
            width, 
            height, 
            tokenizer, 
            label_map 
        )
        
        # 3. Store Result
        all_processed_data.append(aligned_output)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(full_doc_ids)} documents.")

    # 4. Save the final list of dictionaries to disk
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    with open(processed_file, 'wb') as f:
        pickle.dump(all_processed_data, f)
        
    print(f"\n✅ Successfully processed {len(all_processed_data)} documents.")
    print(f"Data saved to: {processed_file}")


if __name__ == '__main__':
    # Initialize the tokenizer component directly
    TOKENIZER = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
    
    # Start the main data processing loop
    process_and_save_data(
        json_dir=JSON_DIR, 
        image_dir=IMAGE_DIR, 
        processed_file=PROCESSED_FILE, 
        tokenizer=TOKENIZER, 
        label_map=LABEL_MAP
    )