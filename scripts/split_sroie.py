from sklearn.model_selection import train_test_split
import pickle
import os

# --- CONFIGURATION ---
SROIE_PROCESSED_FILE = 'data/processed/sroie_aligned_data.pkl'
SROIE_TRAIN_FILE = 'data/processed/sroie_train.pkl'
SROIE_TEST_FILE = 'data/processed/sroie_test.pkl' # This will be the 20% holdout

# Ensure the processed directory exists before saving
os.makedirs('data/processed', exist_ok=True)


# --- 1. LOAD THE FULL PROCESSED DATA ---
print(f"Loading data from: {SROIE_PROCESSED_FILE}")
try:
    with open(SROIE_PROCESSED_FILE, 'rb') as f:
        # Load the list of processed document dictionaries (626 total)
        sroie_data = pickle.load(f)
except FileNotFoundError:
    print("ERROR: Processed data file not found. Ensure prepare_sroie_data.py was run.")
    exit()

print(f"Total documents loaded: {len(sroie_data)}")


# --- 2. PERFORM 80/20 SPLIT ---
# 80% for Training, 20% for Testing (used for validation and final test)
sroie_train, sroie_test = train_test_split(
    sroie_data, 
    test_size=0.20, # 20% for testing (approx. 125 documents)
    random_state=42, # Set for reproducibility
    shuffle=True
)


# --- 3. SAVE THE SPLITS ---
print("\n--- Saving Final Splits (80% Train, 20% Test) ---")

# Save Training Set
with open(SROIE_TRAIN_FILE, 'wb') as f:
    pickle.dump(sroie_train, f)

# Save Testing Set (This 20% will be used for both validation and final testing)
with open(SROIE_TEST_FILE, 'wb') as f:
    pickle.dump(sroie_test, f)


# --- 4. PRINT VERIFICATION ---
print(f"✅ SROIE Training Set saved: {len(sroie_train)} documents ({SROIE_TRAIN_FILE})")
print(f"✅ SROIE Testing/Holdout Set saved: {len(sroie_test)} documents ({SROIE_TEST_FILE})")