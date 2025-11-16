import json
from PIL import Image
import os

doc_id = '71108371'
image_path = os.path.join('data', 'raw', 'funsd','dataset', 'training_data', 'images', f'{doc_id}.png')
json_path = os.path.join('data', 'raw', 'funsd', 'dataset', 'training_data', 'annotations', f'{doc_id}.json')

with Image.open(image_path) as img:
    w_orig, h_orig = img.size
    print(f'image dimensions: w- {w_orig}, h- {h_orig}')

with open(json_path, 'r') as f:
    data=json.load(f)
    print('\nsuccesfully loaded json annotation data')

# --- Continue the script from above ---

print("\n--- Annotation Sample ---")
# Assuming the data structure above:
for i, entity in enumerate(data['form'][:3]): # Print the first 3 entities
    box = entity['box']
    text = entity['text']
    
    print(f"Token: '{text}'")
    print(f"Raw Box: {box}")
    
    # Check if the coordinates are raw pixel values
    X_max_raw = box[2]
    Y_max_raw = box[3]
    print(f"  X_max ({X_max_raw}) vs W_orig ({w_orig})")
    print(f"  Y_max ({Y_max_raw}) vs H_orig ({w_orig})")
    print("-" * 20)
    
    if i == 2: break # Limit output   