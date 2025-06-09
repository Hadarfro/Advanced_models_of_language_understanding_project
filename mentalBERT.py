import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np

# 📥 Load and Balance Your Dataset
df = pd.read_csv("Combined Data.csv")  # replace with your actual file
sample_size = 1000

balanced_df = (
    df.groupby("status", group_keys=False)
    .apply(lambda x: x.sample(n=min(sample_size, len(x)), random_state=42))
    .reset_index(drop=True)
)

# ✏️ Rename columns (if needed)
balanced_df = balanced_df.rename(columns={"status": "label", "text_column_name": "text"})  # replace text_column_name!

# 🔀 Train/Test Split
train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df["label"])

# 🔄 Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 🏷️ Create Label Mappings
labels = sorted(balanced_df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

train_dataset = train_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

# 🧠 Load MentalBERT and Tokenizer
model_name = "mental/mental-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ✂️ Tokenize the Data
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ⚙️ Training Arguments
training_args = TrainingArguments(
    output_dir="./mentalbert_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 📊 Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean()}

# 🧪 Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 🚀 Train the Model
print("\n🚀 Training MentalBERT...")
trainer.train()

# 📈 Evaluate on Test Set
print("\n📊 Predicting...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)]))
