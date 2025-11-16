import json
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Import Functions from normalization.py ---
# This assumes both files are in the same directory (scripts/)
from normalization import normalize_bbox, get_image_dimensions
from label_distribution_analysis import get_all_document_ids


def plot_spatial_clustering(doc_ids, json_dir, image_dir):
    x_centers = []
    y_centers = []
    
    for doc_id in doc_ids:
        # 1. Get original dimensions and load JSON
        width, height = get_image_dimensions(doc_id, image_dir)
        if width == 0 or height == 0: continue
            
        json_path = os.path.join(json_dir, f'{doc_id}.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception: continue
        
        # 2. Process all tokens belonging to meaningful entities
        for entity in data.get('form', []):
            if entity.get('label') in ['question', 'answer']:
                
                # We use the box of the entire entity block for simplicity in EDA
                raw_box = entity['box']
                norm_box = normalize_bbox(raw_box, width, height)
                x0, y0, x1, y1 = norm_box
                
                # 3. Calculate the center point
                x_center = (x0 + x1) / 2
                y_center = (y0 + y1) / 2
                
                x_centers.append(x_center)
                y_centers.append(y_center)
                
    # 4. Plotting
    plt.figure(figsize=(8, 8))
    
    # Plotting 1000 - Yc: Flips the Y-axis so 0 is at the bottom (standard math plot view)
    plt.scatter(x_centers, 1000 - np.array(y_centers), s=10, alpha=0.6, color='darkviolet')
    
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.title('Spatial Clustering of Key Entities (Normalized 1000x1000 Grid)')
    plt.xlabel('Normalized X Coordinate (0 - 1000)')
    plt.ylabel('Normalized Y Coordinate (0 - 1000) [Bottom to Top]')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.savefig('eda_spatial_clustering.png')
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    json_dir = 'data/raw/funsd/dataset/training_data/annotations'
    image_dir = 'data/raw/funsd/dataset/training_data/images' 
    full_doc_ids = get_all_document_ids(json_dir)
    plot_spatial_clustering(full_doc_ids, json_dir, image_dir)