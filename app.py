import os
import sys
import json
import torch
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import collections
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

# 1. DEFINE CORE APP PATHS
# APP_ROOT to find the models/ and templates/ folders.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. ALL HELPER FUNCTIONS

# Normalization Functions
def normalize_single_coord(coord_value, original_dimension):
    """Scales a single raw pixel coordinate (X or Y) to the 0-1000 range."""
    if original_dimension == 0:
        return 0
    
    # Formula: round( (Raw / Original_Dimension) * 1000 )
    normalized = round((coord_value / original_dimension) * 1000)
    
    # Clamping: ensures value is strictly between 0 and 1000.
    return max(0, min(1000, normalized))

def normalize_bbox(bbox, width, height):
    """Normalizes a full bounding box [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = bbox
    
    # Normalize X coordinates using Width
    x0_norm = normalize_single_coord(x0, width)
    x1_norm = normalize_single_coord(x1, width)
    
    # Normalize Y coordinates using Height
    y0_norm = normalize_single_coord(y0, height)
    y1_norm = normalize_single_coord(y1, height)
    
    return [x0_norm, y0_norm, x1_norm, y1_norm]

# Inference Functions
def preprocess_image(image_path):
    """Loads an image, runs OCR, and returns normalized tokens/boxes."""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    print(f"Running Tesseract OCR on image (Size: {width}x{height})...")
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    
    # Filter OCR results
    ocr_data = ocr_data[ocr_data.conf > 50] 
    ocr_data = ocr_data[ocr_data.text.notna() & (ocr_data.text.str.strip() != '')]

    tokens = []
    normalized_boxes = []

    for _, row in ocr_data.iterrows():
        text = str(row['text'])
        x, y, w, h = row['left'], row['top'], row['width'], row['height']
        raw_box = [x, y, x + w, y + h]
        normalized_box = normalize_bbox(raw_box, width, height)
        
        tokens.append(text)
        normalized_boxes.append(normalized_box)
        
    return tokens, normalized_boxes

def stitch_entities(tokens, boxes, predictions, id_to_label):
    """Combines B- and I- tags into full entities."""
    entity_lists = collections.defaultdict(list)
    current_entity_tokens = []
    current_entity_boxes = []
    current_entity_tag = "O"

    for token, box, pred_id in zip(tokens, boxes, predictions):
        tag = id_to_label.get(pred_id, "O")
        
        if tag.startswith("B-"):
            # If we are starting a new entity, save the previous one
            if current_entity_tokens:
                label = current_entity_tag.replace("B-", "")
                entity_lists[label].append({
                    "text": " ".join(current_entity_tokens),
                    "box": np.mean(current_entity_boxes, axis=0).tolist()
                })
            # Start the new entity
            current_entity_tag = tag
            current_entity_tokens = [token]
            current_entity_boxes = [box]
        
        elif tag.startswith("I-") and tag.replace("I-", "") == current_entity_tag.replace("B-", ""):
            # Continue the current entity
            current_entity_tokens.append(token)
            current_entity_boxes.append(box)
        else: 
            # If it's an "O" tag or a mismatched "I-" tag, close the current entity
            if current_entity_tokens:
                label = current_entity_tag.replace("B-", "")
                entity_lists[label].append({
                    "text": " ".join(current_entity_tokens),
                    "box": np.mean(current_entity_boxes, axis=0).tolist()
                })
            current_entity_tag = "O"
            current_entity_tokens = []
            current_entity_boxes = []

    # Save the very last entity
    if current_entity_tokens:
        label = current_entity_tag.replace("B-", "")
        entity_lists[label].append({
            "text": " ".join(current_entity_tokens),
            "box": np.mean(current_entity_boxes, axis=0).tolist()
        })
        
    return entity_lists

def link_keys_and_values(stitched_entities, id_to_label):
    """
    Applies spatial linking logic to find key-value pairs.
    """
    keys = stitched_entities.get('FIELD_KEY', [])
    values = stitched_entities.get('FIELD_VALUE', [])
    
    # Store other entities (headers, totals, etc.)
    other_entities = {}
    for label, entities in stitched_entities.items():
        if label not in ['FIELD_KEY', 'FIELD_VALUE']:
            other_entities[label.lower()] = " | ".join([e['text'] for e in entities])

    unlinked_values = list(values)
    linked_pairs = {}

    for key in keys:
        key_y_center = (key['box'][1] + key['box'][3]) / 2
        key_x_end = key['box'][2]
        
        best_match = None
        min_distance = float('inf')
        best_match_index = -1
        
        # Find the closest value on the same horizontal line
        for i, value in enumerate(unlinked_values):
            value_y_center = (value['box'][1] + value['box'][3]) / 2
            value_x_start = value['box'][0]
            
            # Check for vertical alignment (within 10 pixels on a 1000-scale)
            if abs(key_y_center - value_y_center) < 10: 
                # Check for horizontal proximity (value is to the right of the key)
                horizontal_distance = value_x_start - key_x_end
                if horizontal_distance > 0 and horizontal_distance < min_distance:
                    min_distance = horizontal_distance
                    best_match = value['text']
                    best_match_index = i
            
        if best_match is not None:
            clean_key = key['text'].replace(":", "").strip()
            linked_pairs[clean_key] = best_match
            unlinked_values.pop(best_match_index) # Value is now linked, remove it
            
    # Compile the final JSON output
    final_output = other_entities
    final_output["linked_key_values"] = linked_pairs
    if unlinked_values:
        final_output["_unlinked_values"] = [v['text'] for v in unlinked_values]

    return final_output

def run_inference(image_path, model, tokenizer, device, label_map, id_to_label):
    """
    Main function to run the full pipeline on a single image.
    """
    tokens, normalized_boxes = preprocess_image(image_path)
    if not tokens:
        return {"error": "No text detected or image could not be loaded."}

    # Create dummy labels for inference
    word_labels = [label_map['O']] * len(tokens) 
    
    # Tokenize the words and boxes
    encoding = tokenizer(
        tokens, 
        boxes=normalized_boxes, 
        word_labels=word_labels,
        truncation=True,
        padding="max_length", 
        return_tensors="pt"
    )

    with torch.no_grad():
        # Move inputs to the correct device (GPU or CPU)
        inputs = {k: v.to(device) for k, v in encoding.items() if k != 'labels'}
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    
    # Align predictions back to original words
    word_ids = encoding.word_ids()
    predicted_labels_for_words = []
    current_word_id = -1
    
    for token_idx, pred_id in enumerate(predictions):
        word_id = word_ids[token_idx]
        if word_id is None: continue # Skip [CLS] and [SEP] tokens
        if pred_id == -100: continue # Skip padding tokens
        
        if word_id != current_word_id: 
            # This is the first token of a new word
            current_word_id = word_id
            predicted_labels_for_words.append(pred_id)
            
    # Handle potential truncation if tokens > 512
    if len(predicted_labels_for_words) < len(tokens):
        predicted_labels_for_words.extend([label_map['O']] * (len(tokens) - len(predicted_labels_for_words)))
    
    # Stitch B-I-O tags together
    stitched_entities = stitch_entities(tokens, normalized_boxes, predicted_labels_for_words, id_to_label)
    
    # Run spatial linking
    final_json = link_keys_and_values(stitched_entities, id_to_label)
    
    return final_json

# --- 3. CONFIGURATION AND MODEL LOADING (GLOBAL) ---
MODEL_PATH = os.path.join(APP_ROOT, "models", "documind_layoutlmv3_best")
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define Label Maps
LABEL_MAP = {
    "O": 0, "B-TOTAL_AMOUNT": 1, "I-TOTAL_AMOUNT": 2,
    "B-INVOICE_DATE": 3, "I-INVOICE_DATE": 4, "B-VENDOR_NAME": 5, "I-VENDOR_NAME": 6,
    "B-VENDOR_ADDRESS": 7, "I-VENDOR_ADDRESS": 8, "B-FIELD_KEY": 9, "I-FIELD_KEY": 10,
    "B-FIELD_VALUE": 11, "I-FIELD_VALUE": 12, "B-DOCUMENT_ID": 13, "I-DOCUMENT_ID": 14,
    "B-LINE_ITEM": 15, "I-LINE_ITEM": 16,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
ID_TO_LABEL[-100] = "PAD" # For padding tokens

# Load Model and Tokenizer (globally, once at startup)
print(f"Loading model from {MODEL_PATH}...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
MODEL.to(DEVICE)
MODEL.eval() 
print(f"Model loaded successfully on device: {DEVICE}")


# --- 4. FLASK APPLICATION ---
app = Flask(__name__, template_folder='templates') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Serves the main HTML page (index.html)"""
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def extract_data():
    """API endpoint to handle file upload and extraction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        
        # Create a unique filename to prevent conflicts
        unique_filename = f"{uuid.uuid4()}{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)

        image_to_process = save_path
        
        # --- PDF to Image Conversion ---
        if ext == '.pdf':
            print(f"Converting PDF: {save_path}")
            try:
                # poppler-utils must be installed on the system
                images = convert_from_path(save_path)
                if images:
                    # Use the first page of the PDF
                    image_to_process = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.png")
                    images[0].save(image_to_process, 'PNG')
                else:
                    return jsonify({"error": "Could not convert PDF to image"}), 500
            except Exception as e:
                return jsonify({"error": f"PDF conversion failed (is Poppler installed?): {e}"}), 500

        # Run Inference
        print(f"Running inference on: {image_to_process}")
        try:
            results = run_inference(
                image_to_process, 
                MODEL, 
                TOKENIZER, 
                DEVICE, 
                LABEL_MAP, 
                ID_TO_LABEL
            )
            
            return jsonify(results)
        
        except Exception as e:
            return jsonify({"error": f"Inference failed: {e}"}), 500
        
        finally:
            # Clean up temporary files
            if os.path.exists(save_path):
                os.remove(save_path)
            if image_to_process != save_path and os.path.exists(image_to_process):
                # This removes the temporary PNG if a PDF was uploaded
                os.remove(image_to_process)

if __name__ == '__main__':
    # Get port from environment variable, default to 8080 (required for Cloud Run)
    port = int(os.environ.get("PORT", 8080))
    # Run on 0.0.0.0 to be accessible externally (required for Cloud Run)
    app.run(debug=False, host='0.0.0.0', port=port)