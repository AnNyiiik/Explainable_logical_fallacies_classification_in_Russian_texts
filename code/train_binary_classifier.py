#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import gc
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import GradScaler, autocast   

with open("./all_pairs.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)   # список объектов с ключами original_text, neutral_text, original_label

try:
    with open("./extra_fallacies.json", "r", encoding="utf-8") as f:
        extra_fall = json.load(f)
    if isinstance(extra_fall, list) and len(extra_fall) > 0 and isinstance(extra_fall[0], dict):
        extra_fall = [item["text"] for item in extra_fall]
except FileNotFoundError:
    extra_fall = []
    print("Файл extra_fallacies.json не найден, пропускаем.")

try:
    with open("./extra_neutrals.json", "r", encoding="utf-8") as f:
        extra_neut = json.load(f)
    if isinstance(extra_neut, list) and len(extra_neut) > 0 and isinstance(extra_neut[0], dict):
        extra_neut = [item["text"] for item in extra_neut]
except FileNotFoundError:
    extra_neut = []
    print("Файл extra_neutrals.json не найден, пропускаем.")

texts = []
binary_labels = []

for p in pairs:
    texts.append(p["original_text"])
    binary_labels.append(1)

for p in pairs:
    texts.append(p["neutral_text"])
    binary_labels.append(0)

for t in extra_fall:
    texts.append(t)
    binary_labels.append(1)

for t in extra_neut:
    texts.append(t)
    binary_labels.append(0)

print(f"Всего примеров: {len(texts)}")
print(f"Доля ошибок: {sum(binary_labels)/len(binary_labels):.3f}")

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, binary_labels, test_size=0.3, random_state=42, stratify=binary_labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train: {len(train_texts)} (pos: {sum(train_labels)})")
print(f"Val:   {len(val_texts)} (pos: {sum(val_labels)})")
print(f"Test:  {len(test_texts)} (pos: {sum(test_labels)})")

class BinaryDataset(TorchDataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)

batch_size = 2
accumulation_steps = 8   # effective batch = 16
train_loader = DataLoader(BinaryDataset(train_texts, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(BinaryDataset(val_texts, val_labels), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(BinaryDataset(test_texts, test_labels), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class TrainableBinaryClassifier(nn.Module):
    def __init__(self, encoder_model_name, num_classes=2):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_model_name)
        self.encoder.max_seq_length = 512
        embed_dim = self.encoder.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, sentences):
        # Токенизация
        inputs = self.encoder.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.encoder.max_seq_length)
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        transformer = self.encoder._first_module().auto_model
        outputs = transformer(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]   # [CLS]
        return self.classifier(embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_name = "user-bge-contrastive-finetuned"   # или "deepvk/USER-bge-m3"
model = TrainableBinaryClassifier(encoder_name, num_classes=2).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=3)
scaler = GradScaler()

epochs = 5
best_val_loss = float('inf')
best_model_path = "best_binary_classifier_full.pt"

for epoch in range(epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (texts, labels) in enumerate(train_loader):
        texts = list(texts)
        labels = labels.to(device)
        with autocast():
            logits = model(texts)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    scheduler.step()

    # Валидация
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts = list(texts)
            labels = labels.to(device)
            logits = model(texts)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> Saved best model (val loss {avg_val_loss:.4f})")

model.load_state_dict(torch.load(best_model_path))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        texts = list(texts)
        labels = labels.to(device)
        logits = model(texts)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
print("\n=== Test set results ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1_macro:.4f}")
print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=["neutral", "fallacy"]))