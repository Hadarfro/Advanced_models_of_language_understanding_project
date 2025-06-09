from transformers import (
    GPT2Tokenizer, GPT2ForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import torch
import numpy as np

# Load dataset
df = pd.read_csv("Combined Data.csv")

# Balance the dataset
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

# Split dataset
train_df, test_df = train_test_split(balanced_subset, test_size=0.2, random_state=42, stratify=balanced_subset["label"])

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenization function
def tokenize_data(example, tokenizer):
    texts = example["statement"]
    texts = [str(t) if t is not None and str(t) != 'nan' else "" for t in texts]
    tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = example["label"]
    return tokens

# Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Load tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label2id))

# Update label mapping
roberta_model.config.label2id = label2id
roberta_model.config.id2label = id2label

# Tokenize datasets
roberta_train = train_dataset.map(lambda x: tokenize_data(x, roberta_tokenizer), batched=True)
roberta_test = test_dataset.map(lambda x: tokenize_data(x, roberta_tokenizer), batched=True)

# Training arguments
roberta_args = TrainingArguments(
    output_dir="./roberta_results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    logging_dir="./logs/roberta",
    evaluation_strategy="epoch"
)

# Trainer
roberta_trainer = Trainer(
    model=roberta_model,
    args=roberta_args,
    train_dataset=roberta_train,
    eval_dataset=roberta_test,
    compute_metrics=compute_metrics,
)

# Train and evaluate
print("\n🚀 Training RoBERTa...")
roberta_trainer.train()
roberta_results = roberta_trainer.predict(roberta_test)

# Report
roberta_y_pred = np.argmax(roberta_results.predictions, axis=1)
roberta_y_true = roberta_results.label_ids

print("\n📊 RoBERTa Test Metrics:", roberta_results.metrics)
print("\n📋 RoBERTa Classification Report:")
print(classification_report(roberta_y_true, roberta_y_pred, target_names=[id2label[i] for i in sorted(id2label.keys())]))
