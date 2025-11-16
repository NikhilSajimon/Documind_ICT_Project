import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
import numpy as np 

# ===============================================================
# 🏷 LABEL MAPS
# ===============================================================
LABEL_MAP = {
    "O": 0, "B-TOTAL_AMOUNT": 1, "I-TOTAL_AMOUNT": 2,
    "B-INVOICE_DATE": 3, "I-INVOICE_DATE": 4,
    "B-VENDOR_NAME": 5, "I-VENDOR_NAME": 6,
    "B-VENDOR_ADDRESS": 7, "I-VENDOR_ADDRESS": 8,
    "B-FIELD_KEY": 9, "I-FIELD_KEY": 10,
    "B-FIELD_VALUE": 11, "I-FIELD_VALUE": 12,
    "B-DOCUMENT_ID": 13, "I-DOCUMENT_ID": 14,
    "B-LINE_ITEM": 15, "I-LINE_ITEM": 16,
}
id_to_label = {v: k for k, v in LABEL_MAP.items()}
id_to_label[-100] = "PAD"

# ===============================================================
# CONFIGURATION (Local Windows Paths)
# ===============================================================
BASE_PROJECT_PATH = r'C:\Users\nikhi\documind-ai'
FUNSD_TRAIN_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'processed', 'funsd_train_aligned_data.pkl')
SROIE_TRAIN_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'processed', 'sroie_train.pkl')
FUNSD_TEST_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'processed', 'funsd_test_aligned_data.pkl')
SROIE_TEST_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'processed', 'sroie_test.pkl')

# --- MODIFICATION: Define a single save directory ---
FINAL_MODEL_DIR = os.path.join(BASE_PROJECT_PATH, 'models', 'documind_generalized_best')
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# ===============================================================
# DATA HELPERS
# ===============================================================
def load_pickle(file_path):
    """Load a .pkl dataset safely."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"Loaded {len(data)} documents from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

class DocuMindDataset(Dataset):
    """PyTorch dataset class for LayoutLM-style tokenized data."""
    def __init__(self, processed_data_list):
        self.data = processed_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "bbox": torch.tensor(sample["bbox"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }

# ===============================================================
# ⚙ TRAINING FUNCTION
# ===============================================================
def train_model(train_data, val_data, model_name, save_dir, lr=5e-5, num_epochs=15, stage="Training"):
    print(f"\n===== Starting {stage.upper()} ({len(train_data)} docs) =====")
    
    # --- MODIFICATION: Set Batch Size to 2 for local RTX 2050 ---
    BATCH_SIZE = 2
    
    train_dataset = DocuMindDataset(train_data)
    validation_dataset = DocuMindDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # NOTE: Using weighted loss is a good idea for combined datasets.
    # We will assume the 17 classes from the map (0-16).
    # This tensor must have 17 items.
    class_weights = torch.tensor([
        1.0,  # O
        3.0, 3.0,  # TOTAL_AMOUNT
        2.5, 2.5,  # INVOICE_DATE
        2.0, 2.0,  # VENDOR_NAME
        2.0, 2.0,  # VENDOR_ADDRESS
        1.0, 1.0,  # FIELD_KEY
        1.0, 1.0,  # FIELD_VALUE
        1.0, 1.0,  # DOCUMENT_ID
        1.0, 1.0   # LINE_ITEM
    ]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(LABEL_MAP))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    metric = evaluate.load("seqeval")

    best_f1, patience_counter, patience = 0.0, 0, 3
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            try:
                outputs = model(**batch)
                logits = outputs.logits.view(-1, len(LABEL_MAP))
                labels = batch["labels"].view(-1)
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
                
            except torch.cuda.OutOfMemoryError:
                print("\nCUDA Out of Memory on this batch. Skipping...")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

        avg_train_loss = total_loss / len(train_loader)
        print(f"Avg Train Loss: {avg_train_loss:.4f}")

        # ===== VALIDATION =====
        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch["labels"]

                for i in range(labels.size(0)):
                    pred_tags, true_tags = [], []
                    for j in range(labels.size(1)):
                        if labels[i][j] != -100:
                            pred_tags.append(id_to_label[predictions[i][j].item()])
                            true_tags.append(id_to_label[labels[i][j].item()])
                    preds.append(pred_tags)
                    refs.append(true_tags)

        metrics = metric.compute(predictions=preds, references=refs, zero_division=0)
        current_f1 = metrics["overall_f1"]
        print(f"Epoch {epoch+1} | Acc: {metrics['overall_accuracy']:.4f} | F1: {current_f1:.4f}")

        # --- Save Best Model ---
        if current_f1 > best_f1:
            print(f"New Best F1: {current_f1:.4f} → Saving model to {save_dir}")
            best_f1 = current_f1
            patience_counter = 0
            model.save_pretrained(save_dir)
        else:
            patience_counter += 1
            print(f"F1 did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"\n{stage.upper()} Training Done | Best F1 = {best_f1:.4f}")
    return best_f1

# ===============================================================
# MAIN PIPELINE
# ===============================================================
if __name__ == "__main__":
    TOKENIZER = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

    funsd_train = load_pickle(FUNSD_TRAIN_FILE)
    sroie_train = load_pickle(SROIE_TRAIN_FILE)
    funsd_test = load_pickle(FUNSD_TEST_FILE)
    sroie_test = load_pickle(SROIE_TEST_FILE)

    if not (funsd_train and sroie_train):
        print("Missing training data. Please ensure FUNSD & SROIE .pkl files exist.")
        exit()

    # --- THIS IS THE KEY CHANGE ---
    # Create one single, mixed "mega-dataset" for training
    combined_train_data = funsd_train + sroie_train
    
    # Create one single, mixed validation set
    combined_validation_data = funsd_test + sroie_test
    
    print("\n--- Data Unification Complete ---")
    print(f"Total Combined Training Documents: {len(combined_train_data)}")
    print(f"Total Combined Validation Documents: {len(combined_validation_data)}")
    # --- END OF CHANGE ---


    # === Run ONE Generalized Training Job ===
    final_f1 = train_model(
        combined_train_data, 
        combined_validation_data,
        model_name="microsoft/layoutlmv3-base", # Start from the base model
        save_dir=FINAL_MODEL_DIR,
        lr=5e-5, 
        num_epochs=15, # Train for 15 epochs
        stage="Generalized Training"
    )

    # --- Save the tokenizer with the final model ---
    TOKENIZER.save_pretrained(FINAL_MODEL_DIR)
    
    print("\n Full Generalized Training Complete.")
    print(f" Final Combined F1: {final_f1:.4f}")
    print(f" Best Generalized Model & Tokenizer Saved → {FINAL_MODEL_DIR}")