from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import torch
import numpy as np

# Load and balance the dataset
df = pd.read_csv("Combined Data.csv")

sample_size = 500
balanced_subset = (
    df.groupby("status", group_keys=False)
    .apply(lambda x: x.sample(n=min(sample_size, len(x)), random_state=42), include_groups=True)
    .reset_index(drop=True)
)

# Encode labels
label2id = {label: idx for idx, label in enumerate(balanced_subset["status"].unique())}
id2label = {v: k for k, v in label2id.items()}
balanced_subset["label"] = balanced_subset["status"].map(label2id)

# Train-test split
train_df, test_df = train_test_split(balanced_subset, test_size=0.2, random_state=42, stratify=balanced_subset["label"])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id))
model.config.label2id = label2id
model.config.id2label = id2label

# Tokenization function
def tokenize(batch):
    texts = batch["statement"]
    texts = [str(t) if t is not None and str(t) != 'nan' else "" for t in texts]
    tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = batch["label"]
    return tokens

# Tokenize
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Training args
training_args = TrainingArguments(
    output_dir="./bert_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    # evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
results = trainer.predict(test_dataset)

# Print results
print("Test Metrics:", results.metrics)
y_pred = np.argmax(results.predictions, axis=1)
y_true = results.label_ids
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label.keys())]))

# Inference example
text = "I'm stressed from the test tomorrow"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
outputs = model(**inputs)
predicted_label = torch.argmax(outputs.logits).item()
print("\nרמת חרדה חוזה:", id2label[predicted_label])
