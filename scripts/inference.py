# --- scripts/inference.py ---
import torch
import pytesseract
from PIL import Image
import numpy as np
import collections

# Import your helper functions from normalization.py
try:
    from normalization import normalize_bbox
except ImportError:
    print("Error: normalization.py not found.")
    def normalize_bbox(bbox, width, height): return bbox

# --- 1. PRE-PROCESSING (OCR AND NORMALIZATION) ---
# (This function is unchanged)
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    print(f"Running Tesseract OCR on image (Size: {width}x{height})...")
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    
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

# --- 2. POST-PROCESSING (B-I-O STITCHING & LINKING) ---
# (These functions are also unchanged, but we must pass ID_TO_LABEL)
def stitch_entities(tokens, boxes, predictions, id_to_label):
    entity_lists = collections.defaultdict(list)
    current_entity_tokens = []
    current_entity_boxes = []
    current_entity_tag = "O"

    for token, box, pred_id in zip(tokens, boxes, predictions):
        tag = id_to_label.get(pred_id, "O")
        
        if tag.startswith("B-"):
            if current_entity_tokens:
                label = current_entity_tag.replace("B-", "")
                entity_lists[label].append({
                    "text": " ".join(current_entity_tokens),
                    "box": np.mean(current_entity_boxes, axis=0).tolist()
                })
            current_entity_tag = tag
            current_entity_tokens = [token]
            current_entity_boxes = [box]
        
        elif tag.startswith("I-") and tag.replace("I-", "") == current_entity_tag.replace("B-", ""):
            current_entity_tokens.append(token)
            current_entity_boxes.append(box)
        else: 
            if current_entity_tokens:
                label = current_entity_tag.replace("B-", "")
                entity_lists[label].append({
                    "text": " ".join(current_entity_tokens),
                    "box": np.mean(current_entity_boxes, axis=0).tolist()
                })
            current_entity_tag = "O"
            current_entity_tokens = []
            current_entity_boxes = []

    if current_entity_tokens:
        label = current_entity_tag.replace("B-", "")
        entity_lists[label].append({
            "text": " ".join(current_entity_tokens),
            "box": np.mean(current_entity_boxes, axis=0).tolist()
        })
        
    return entity_lists

def link_keys_and_values(stitched_entities, id_to_label):
    keys = stitched_entities.get('FIELD_KEY', [])
    values = stitched_entities.get('FIELD_VALUE', [])
    
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
        
        for i, value in enumerate(unlinked_values):
            value_y_center = (value['box'][1] + value['box'][3]) / 2
            value_x_start = value['box'][0]
            
            if abs(key_y_center - value_y_center) < 10: 
                horizontal_distance = value_x_start - key_x_end
                if horizontal_distance > 0 and horizontal_distance < min_distance:
                    min_distance = horizontal_distance
                    best_match = value['text']
                    best_match_index = i
        
        if best_match is not None:
            clean_key = key['text'].replace(":", "").strip()
            linked_pairs[clean_key] = best_match
            unlinked_values.pop(best_match_index)
            
    final_output = other_entities
    final_output["linked_key_values"] = linked_pairs
    if unlinked_values:
        final_output["_unlinked_values"] = [v['text'] for v in unlinked_values]

    return final_output

# --- 3. MAIN INFERENCE FUNCTION ---
def run_inference(image_path, model, tokenizer, device, label_map, id_to_label):
    """
    Main function to run the full pipeline on a single image.
    NOW ACCEPTS THE MODEL AND TOKENIZER AS ARGUMENTS.
    """
    tokens, normalized_boxes = preprocess_image(image_path)
    if not tokens:
        return {"error": "No text detected or image could not be loaded."}

    word_labels = [label_map['O']] * len(tokens) 
    
    encoding = tokenizer(
        tokens, 
        boxes=normalized_boxes, 
        word_labels=word_labels,
        truncation=True,
        padding="max_length", 
        return_tensors="pt"
    )

    with torch.no_grad():
        # Move inputs to the correct device (GPU)
        inputs = {k: v.to(device) for k, v in encoding.items() if k != 'labels'}
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    
    word_ids = encoding.word_ids()
    predicted_labels_for_words = []
    current_word_id = -1
    
    for token_idx, pred_id in enumerate(predictions):
        word_id = word_ids[token_idx]
        if word_id is None: continue
        if pred_id == -100: continue
        if word_id != current_word_id: 
            current_word_id = word_id
            predicted_labels_for_words.append(pred_id)
            
    if len(predicted_labels_for_words) < len(tokens):
        predicted_labels_for_words.extend([label_map['O']] * (len(tokens) - len(predicted_labels_for_words)))
    
    stitched_entities = stitch_entities(tokens, normalized_boxes, predicted_labels_for_words, id_to_label)
    final_json = link_keys_and_values(stitched_entities, id_to_label)
    
    return final_json
