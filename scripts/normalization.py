import math
from PIL import Image
import os
import json
import numpy as np

# --- 1. CORE SCALING FUNCTION ---
def normalize_single_coord(coord_value, original_dimension):
    """Scales a single raw pixel coordinate (X or Y) to the 0-1000 range."""
    if original_dimension == 0:
        return 0
    
    # Formula: round( (Raw / Original_Dimension) * 1000 )
    normalized = round((coord_value / original_dimension) * 1000)
    
    # Clamping: ensures value is strictly between 0 and 1000.
    return max(0, min(1000, normalized))

# --- 2. FULL BOUNDING BOX NORMALIZATION ---
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

# --- 3. IMAGE DIMENSION RETRIEVAL (Helper) ---
def get_image_dimensions(doc_id, image_dir='data/raw/funsd/dataset/training_data/images'):
    """Gets the original pixel width and height of an image."""
    # Note: Assumes FUNSD images are PNG
    image_path = os.path.join(image_dir, f'{doc_id}.png')
    
    try:
        with Image.open(image_path) as img:
            return img.size # Returns (width, height)
    except FileNotFoundError:
        return 0, 0
    


# Assuming the three functions (normalize_single_coord, normalize_bbox, get_image_dimensions)
# are defined above or imported correctly.

# --- DEMONSTRATION BLOCK ---
if __name__ == "__main__":
    # 1. Define a sample document's original size
    # Scenario 1: A wide document (e.g., a landscape scan)
    WIDTH_A = 1200 
    HEIGHT_A = 800

    # 2. Define a raw bounding box in the center-right of the document
    # Raw Box: [Xmin, Ymin, Xmax, Ymax]
    RAW_BOX_A = [800, 300, 1050, 450] 


    print(f"--- Scenario A: Wide Document ({WIDTH_A}x{HEIGHT_A}) ---")
    print(f"Raw Box (Pixels): {RAW_BOX_A}")

    # --- Execute the normalization function ---
    normalized_box_A = normalize_bbox(RAW_BOX_A, WIDTH_A, HEIGHT_A)

    print(f"Normalized Box (0-1000): {normalized_box_A}")

    print("-" * 50)

    # Scenario 2: A square document (e.g., a receipt section)
    WIDTH_B = 1000 
    HEIGHT_B = 1000
    RAW_BOX_B = [50, 50, 250, 250] # A box in the top-left corner

    print(f"--- Scenario B: Square Document ({WIDTH_B}x{HEIGHT_B}) ---")
    print(f"Raw Box (Pixels): {RAW_BOX_B}")

    normalized_box_B = normalize_bbox(RAW_BOX_B, WIDTH_B, HEIGHT_B)

    print(f"Normalized Box (0-1000): {normalized_box_B}")

    # --- Demonstration of Single Coordinate Scaling ---
    print("-" * 50)
    test_coord = 800
    test_dim = 1200
    norm_val = normalize_single_coord(test_coord, test_dim)
    print(f"Test Single Coord: {test_coord}/{test_dim} scaled to: {norm_val}")