import pandas as pd
import os
from normalization import get_image_dimensions
from label_distribution_analysis import get_all_document_ids 

def generate_layout_variance_table(doc_ids, image_dir='data/raw/funsd/dataset/training_data/images', csv_filename='layout_variance_report.csv'):
    """
    Retrieves the original pixel dimensions (W x H) for a list of documents
    to demonstrate layout variability, justifying normalization.
    """
    variance_data = []
    
    for doc_id in doc_ids:
        # Calls the helper function to get (width, height)
        width, height = get_image_dimensions(doc_id, image_dir) 
        
        # Calculate Aspect Ratio (W/H) as an additional metric
        aspect_ratio = round(width / height, 2) if height else 0
        
        variance_data.append({
            'Document ID': doc_id,
            'Original Width (W)': width,
            'Original Height (H)': height,
            'Aspect Ratio (W/H)': aspect_ratio
        })

    # Create a DataFrame for professional tabular output
    df = pd.DataFrame(variance_data)
    
    # Calculate min/max range for the report summary
    min_w = df['Original Width (W)'].min()
    max_w = df['Original Width (W)'].max()
    
    print("\n--- Document Layout Variance Report (Justifies Normalization) ---")
    print(f"Width Range: {min_w} to {max_w} pixels")
    print("-" * 65)
    # Use .to_string to print the full DataFrame without index
    print(df.to_string(index=False)) 
    print("-" * 65)
    
    # --- ADDED CODE: Save DataFrame to CSV ---
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Results saved to {csv_filename}")
    
    return df

# --- EXECUTION ---
if __name__ == "__main__":
    JSON_DIR = 'data/raw/funsd/dataset/training_data/annotations'
    IMAGE_DIR = 'data/raw/funsd/dataset/training_data/images' 
    
    # Get the list of all IDs to process (using the imported function)
    full_doc_ids = get_all_document_ids(JSON_DIR) 
    
    # Run the analysis and save the CSV
    # We pass the desired filename here
    variance_df = generate_layout_variance_table(
        doc_ids=full_doc_ids, 
        image_dir=IMAGE_DIR, 
        csv_filename='layout_variance_report.csv'
    )