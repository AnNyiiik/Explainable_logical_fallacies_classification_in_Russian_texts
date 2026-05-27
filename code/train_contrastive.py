#!/usr/bin/env python3
import json
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open("./all_pairs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences1 = [item["original_text"] for item in data]
sentences2 = [item["neutral_text"] for item in data]

labels_for_split = [item["original_label"] for item in data]
train_idx, val_idx = train_test_split(
    range(len(sentences1)),
    test_size=0.1,
    random_state=42,
    stratify=labels_for_split
)

train_s1 = [sentences1[i] for i in train_idx]
train_s2 = [sentences2[i] for i in train_idx]
val_s1   = [sentences1[i] for i in val_idx]
val_s2   = [sentences2[i] for i in val_idx]

train_dataset = Dataset.from_dict({"sentence1": train_s1, "sentence2": train_s2})
val_dataset   = Dataset.from_dict({"sentence1": val_s1,   "sentence2": val_s2})

print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

model = SentenceTransformer("deepvk/USER-bge-m3")
model.max_seq_length = 512

train_loss = CachedMultipleNegativesRankingLoss(model, scale=20.0)

args = SentenceTransformerTrainingArguments(
    output_dir="user-bge-contrastive",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=train_loss,
)

trainer.train()

model.save_pretrained("user-bge-contrastive-finetuned")
print("Contrastive training complete, model saved.")

print("\nEvaluating quality on validation pairs (Logistic Regression on embeddings):")
val_texts = []
val_binary = []
for i in range(len(val_s1)):
    val_texts.append(val_s1[i]); val_binary.append(1)
    val_texts.append(val_s2[i]); val_binary.append(0)

embeddings = model.encode(val_texts, show_progress_bar=True)
X_train_emb, X_val_emb, y_train_emb, y_val_emb = train_test_split(embeddings, val_binary, test_size=0.5, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train_emb)
acc = accuracy_score(y_val_emb, clf.predict(X_val_emb))
print(f"LogReg accuracy on val pairs (quick check): {acc:.4f}")