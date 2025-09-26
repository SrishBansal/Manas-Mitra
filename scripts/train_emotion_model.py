import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import os

# Determine device for models (use string names for torch.device)
if torch.cuda.is_available():
    DEVICE_STR = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_STR = "mps"
else:
    DEVICE_STR = "cpu"

print(f"Using device: {DEVICE_STR}")

# 1. Load dataset
# Using the new English-only dataset
df = pd.read_csv("lrec_translated_400.csv")
print("Columns after initial CSV read:", df.columns)

# Rename columns for clarity and consistency with the original code structure
# Ensure 'utterance' and 'emotions' exist before renaming
if 'utterance' in df.columns and 'emotions' in df.columns:
    df = df.rename(columns={'utterance': 'Text', 'emotions': 'Label'})
else:
    print("Error: 'utterance' or 'emotions' columns not found in the CSV. Please check your CSV file.")
    exit() # Exit if essential columns are missing

print("Columns after renaming:", df.columns)

# Process 'Label' column (originally 'emotions') to take only the first emotion if multiple are present
df["Label"] = df["Label"].apply(lambda x: str(x).split(',')[0].strip())

# Also convert 'Text' and 'Label' to string to handle potential mixed types
df["Text"] = df["Text"].astype(str)
df["Label"] = df["Label"].astype(str)

# --- Language filtering is no longer needed as the dataset is already translated to English ---
# Instead, ensure that the remaining entries are valid for training.

# Drop NA
df = df.dropna(subset=["Text", "Label"]).reset_index(drop=True)

# Ensure there are still entries after processing and dropping NA
if len(df) == 0:
    print("No valid entries found after processing. Cannot proceed with training.")
    exit()

# Encode labels
labels = df["Label"].unique().tolist()
if not labels:
    print("No unique labels found after processing. Cannot proceed with training.")
    exit()

# Identify and filter out minority classes (labels with only one sample) for stratified split
label_counts = df["Label"].value_counts()
minority_labels = label_counts[label_counts < 2].index.tolist()

if minority_labels:
    print(f"Warning: Removing minority labels for stratified split: {minority_labels}")
    df = df[~df["Label"].isin(minority_labels)].reset_index(drop=True)

# Re-encode labels after filtering minority classes
labels = df["Label"].unique().tolist()
if not labels:
    print("No unique labels found after filtering minority classes. Cannot proceed with training.")
    exit()
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df["label"] = df["Label"].map(label2id)

print(f"Number of unique labels: {len(labels)}")
print(f"Labels mapping: {label2id}")
print(f"Number of English entries after final processing (and minority class removal): {len(df)}")

def stratified_split(df_in: pd.DataFrame, label_col: str, test_size: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    test_parts = []
    train_parts = []
    for lbl, grp in df_in.groupby(label_col):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size)) if len(idx) > 1 else 1
        test_idx = idx[:n_test]
        train_idx = idx[n_test:] if len(idx) > n_test else idx[:0]
        test_parts.append(df_in.loc[test_idx])
        if len(train_idx) > 0:
            train_parts.append(df_in.loc[train_idx])
    test_out = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_out = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_out, test_out

# 3. Convert to HuggingFace dataset
# Ensure there are enough samples for stratify after processing
if len(df) < 2 or len(df["label"].unique()) < 2:
    print("Not enough unique labels or samples for stratified split after processing. Skipping training.")
    exit()

train_df, test_df = stratified_split(df, label_col="label", test_size=0.2, seed=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Remove __index_level_0__ column if it exists, as it's not needed for training
if "__index_level_0__" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])
if "__index_level_0__" in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(["__index_level_0__"])

# 4. Tokenize
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["Text"], truncation=True, padding=True, max_length=512)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 5. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
).to(DEVICE_STR)  # Move model to device

# 6. Training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics for evaluation without scikit-learn
def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())

def _weighted_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    # Compute per-class precision/recall/f1 and weight by support
    f1_sum = 0.0
    total = 0
    for c in range(num_classes):
        tp = int(((predictions == c) & (labels == c)).sum())
        fp = int(((predictions == c) & (labels != c)).sum())
        fn = int(((predictions != c) & (labels == c)).sum())
        support = int((labels == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1 * support
        total += support
    return float(f1_sum / total) if total > 0 else 0.0

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = _accuracy(predictions, labels)
    f1 = _weighted_f1(predictions, labels, num_classes=len(set(labels.tolist())))
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to=[],  # Disable reporting to external services
    dataloader_drop_last=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, # Add compute_metrics
)

trainer.train()

# 7. Save model
output_model_dir = "./fine_tuned_emotion_model"
os.makedirs(output_model_dir, exist_ok=True)
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model saved to {output_model_dir}")

# 8. Test in terminal
print("\n--- Testing the fine-tuned model ---")
model.eval()  # Set model to evaluation mode
device_name = DEVICE_STR

while True:
    text = input("Enter a sentence (or 'quit' to exit): ")
    if text.lower() == "quit":
        break
    inputs = tokenizer(text, return_tensors="pt").to(device_name)  # Move inputs to device
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        print("Predicted Emotion:", id2label[prediction])
