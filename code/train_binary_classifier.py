#!/usr/bin/env python3
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a binary fallacy / neutral classifier on top of a sentence encoder."
    )
    parser.add_argument('-pairs_file', type=str, default='./all_pairs.json',
                        help='JSON list of {original_text, neutral_text, original_label}.')
    parser.add_argument('-extra_fallacies_file', type=str, default='./extra_fallacies.json',
                        help='Optional JSON of extra fallacy examples (skipped if missing).')
    parser.add_argument('-extra_neutrals_file', type=str, default='./extra_neutrals.json',
                        help='Optional JSON of extra neutral examples (skipped if missing).')
    parser.add_argument('-encoder_name', type=str, default='user-bge-contrastive-finetuned',
                        help='Encoder to use (fine-tuned dir or e.g. deepvk/USER-bge-m3).')
    parser.add_argument('-best_model_path', type=str, default='best_binary_classifier_full.pt',
                        help='Where to save the best model checkpoint.')
    parser.add_argument('-max_seq_length', type=int, default=512)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective batch = batch_size * this).')
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-seed', type=int, default=42)
    return parser.parse_args()


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


class TrainableBinaryClassifier(nn.Module):
    def __init__(self, encoder_model_name, max_seq_length=512, num_classes=2):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_model_name)
        self.encoder.max_seq_length = max_seq_length
        embed_dim = self.encoder.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, sentences):
        # Tokenization
        inputs = self.encoder.tokenizer(
            sentences, return_tensors="pt", padding=True,
            truncation=True, max_length=self.encoder.max_seq_length
        )
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        transformer = self.encoder._first_module().auto_model
        outputs = transformer(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]   # [CLS]
        return self.classifier(embeddings)


def load_extra(path, kind):
    try:
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f)
        if isinstance(extra, list) and len(extra) > 0 and isinstance(extra[0], dict):
            extra = [item["text"] for item in extra]
        return extra
    except FileNotFoundError:
        print(f"File {path} not found, skipping ({kind}).")
        return []


def main():
    args = parse_args()

    with open(args.pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)   # list of {original_text, neutral_text, original_label}

    extra_fall = load_extra(args.extra_fallacies_file, "extra fallacies")
    extra_neut = load_extra(args.extra_neutrals_file, "extra neutrals")

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

    print(f"Total examples: {len(texts)}")
    print(f"Fallacy ratio: {sum(binary_labels) / len(binary_labels):.3f}")

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, binary_labels, test_size=0.3, random_state=args.seed, stratify=binary_labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )

    print(f"Train: {len(train_texts)} (pos: {sum(train_labels)})")
    print(f"Val:   {len(val_texts)} (pos: {sum(val_labels)})")
    print(f"Test:  {len(test_texts)} (pos: {sum(test_labels)})")

    train_loader = DataLoader(BinaryDataset(train_texts, train_labels),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BinaryDataset(val_texts, val_labels),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(BinaryDataset(test_texts, test_labels),
                             batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainableBinaryClassifier(args.encoder_name, args.max_seq_length, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=3)
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for i, (texts_b, labels) in enumerate(train_loader):
            texts_b = list(texts_b)
            labels = labels.to(device)
            with autocast():
                logits = model(texts_b)
                loss = criterion(logits, labels)
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * args.accumulation_steps
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts_b, labels in val_loader:
                texts_b = list(texts_b)
                labels = labels.to(device)
                logits = model(texts_b)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.best_model_path)
            print(f"  -> Saved best model (val loss {avg_val_loss:.4f})")

    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts_b, labels in test_loader:
            texts_b = list(texts_b)
            labels = labels.to(device)
            logits = model(texts_b)
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


if __name__ == "__main__":
    main()