import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json

# --- Helper to get all document IDs in a directory ---
def get_all_document_ids(json_dir):
    return [f.replace('.json', '') for f in os.listdir(json_dir) if f.endswith('.json')]

def analyze_label_distribution(doc_ids, json_dir='data/raw/funsd/dataset/training_data/annotations'):
    label_counts = defaultdict(int)
    total_tokens = 0
    
    for doc_id in doc_ids:
        json_path = os.path.join(json_dir, f'{doc_id}.json')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception: # Catch FileNot Found and UnicodeDecodeError
            continue
            
        for entity in data.get('form', []):
            label = entity.get('label', 'other') 
            num_tokens = len(entity.get('words', []))
            
            # Grouping for the chart: Meaningful (Q/A) vs. Irrelevant (Header/Other)
            if label in ['question', 'answer']:
                label_counts['Meaningful_Entity'] += num_tokens
            else:
                label_counts['Irrelevant_Context'] += num_tokens
                
            total_tokens += num_tokens
            
    if total_tokens == 0: return # Exit if no data processed

    meaningful_count = label_counts['Meaningful_Entity']
    irrelevant_count = label_counts['Irrelevant_Context']
    
    # Plotting
    data = {
        'Category': ['Meaningful Entities (Q/A)', 'Irrelevant Context (O-Tag)'],
        'Token Count': [meaningful_count, irrelevant_count]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['Category'], df['Token Count'], color=['#4CAF50', '#FF5733'])
    plt.title('Token Distribution: Justification for Class Imbalance Handling')
    plt.ylabel('Total Number of Tokens')

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_tokens) * 100
        plt.text(bar.get_x() + bar.get_width() / 2., height + 50, 
                 f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('eda_label_distribution.png')
    plt.show()

# --- EXECUTION ---

if __name__ == "__main__":
    json_dir = 'data/raw/funsd/dataset/training_data/annotations' 
    full_doc_ids = get_all_document_ids(json_dir)
    analyze_label_distribution(full_doc_ids, json_dir)