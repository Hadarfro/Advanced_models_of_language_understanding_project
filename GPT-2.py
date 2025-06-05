from transformers import (
    GPT2Tokenizer, GPT2ForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
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

# === GPT-2 Setup ===

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("lvwerra/gpt2-imdb")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model = GPT2ForSequenceClassification.from_pretrained("lvwerra/gpt2-imdb", num_labels=len(label2id))
gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
gpt2_model.config.label2id = label2id
gpt2_model.config.id2label = id2label

gpt2_train = train_dataset.map(lambda x: tokenize_data(x, gpt2_tokenizer), batched=True)
gpt2_test = test_dataset.map(lambda x: tokenize_data(x, gpt2_tokenizer), batched=True)

gpt2_args = TrainingArguments(
    output_dir="./gpt2_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs/gpt2",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


gpt2_trainer = Trainer(
    model=gpt2_model,
    args=gpt2_args,
    train_dataset=gpt2_train,
    eval_dataset=gpt2_test,
    compute_metrics=compute_metrics,
)

print("🚀 Training GPT-2...")
gpt2_trainer.train()
gpt2_results = gpt2_trainer.predict(gpt2_test)
print("\n📊 GPT-2 Test Metrics:", gpt2_results.metrics)
gpt2_y_pred = np.argmax(gpt2_results.predictions, axis=1)
gpt2_y_true = gpt2_results.label_ids

# === BERT Setup ===
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id))
bert_model.config.label2id = label2id
bert_model.config.id2label = id2label

bert_train = train_dataset.map(lambda x: tokenize_data(x, bert_tokenizer), batched=True)
bert_test = test_dataset.map(lambda x: tokenize_data(x, bert_tokenizer), batched=True)

bert_args = TrainingArguments(
    output_dir="./bert_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs/bert",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


bert_trainer = Trainer(
    model=bert_model,
    args=bert_args,
    train_dataset=bert_train,
    eval_dataset=bert_test,
    compute_metrics=compute_metrics,
)

print("\n🚀 Training BERT...")
bert_trainer.train()
bert_results = bert_trainer.predict(bert_test)
print("\n📊 BERT Test Metrics:", bert_results.metrics)
bert_y_pred = np.argmax(bert_results.predictions, axis=1)
bert_y_true = bert_results.label_ids

# === Compare Reports ===
print("\n📋 GPT-2 Classification Report:")
print(classification_report(gpt2_y_true, gpt2_y_pred, target_names=[id2label[i] for i in sorted(id2label.keys())]))

print("\n📋 BERT Classification Report:")
print(classification_report(bert_y_true, bert_y_pred, target_names=[id2label[i] for i in sorted(id2label.keys())]))
