#!/usr/bin/env python3
import json
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import GradScaler, autocast
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a binary fallacy / neutral classifier with 5-fold cross-validation."
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
                        help='Where to save the best model trained on all train data.')
    parser.add_argument('-max_seq_length', type=int, default=512)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective batch = batch_size * this).')
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-cv_folds', type=int, default=5,
                        help='Number of folds for cross-validation.')
    parser.add_argument('-test_size', type=float, default=0.15,
                        help='Fraction of data to hold out as final test set (0 = no separate test).')
    parser.add_argument('-save_fold_models', action='store_true',
                        help='Save the best model for each fold (saved as best_model_fold_*.pt).')
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
        inputs = self.encoder.tokenizer(
            sentences, return_tensors="pt", padding=True,
            truncation=True, max_length=self.encoder.max_seq_length
        )
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        transformer = self.encoder._first_module().auto_model
        outputs = transformer(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
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


def train_fold(train_loader, val_loader, model, criterion, optimizer, scheduler,
               scaler, epochs, accumulation_steps, device, fold_model_path=None):
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for i, (texts_b, labels) in enumerate(train_loader):
            texts_b = list(texts_b)
            labels = labels.to(device)
            with autocast():
                logits = model(texts_b)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps

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

        print(f"  Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if fold_model_path:
                torch.save(model.state_dict(), fold_model_path)
                print(f"  -> Saved best model for fold (val loss {avg_val_loss:.4f})")

    return best_state, best_val_loss


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts_b, labels in loader:
            texts_b = list(texts_b)
            labels = labels.to(device)
            logits = model(texts_b)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    with open(args.pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

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

    # Split off a test set if requested
    if args.test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, binary_labels, test_size=args.test_size,
            random_state=args.seed, stratify=binary_labels
        )
        print(f"Test set: {len(X_test)} examples (pos: {sum(y_test)})")
    else:
        X_train, y_train = texts, binary_labels
        X_test, y_test = [], []

    # Cross-validation on X_train, y_train
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fold_val_metrics = []
    fold_test_metrics = []   # will be filled if test set exists
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{args.cv_folds}")
        print(f"{'='*50}")

        train_texts = [X_train[i] for i in train_idx]
        train_labels = [y_train[i] for i in train_idx]
        val_texts = [X_train[i] for i in val_idx]
        val_labels = [y_train[i] for i in val_idx]

        train_loader = DataLoader(
            BinaryDataset(train_texts, train_labels),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            BinaryDataset(val_texts, val_labels),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        model = TrainableBinaryClassifier(args.encoder_name, args.max_seq_length, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=3)
        scaler = GradScaler()

        fold_model_path = None
        if args.save_fold_models:
            os.makedirs("cv_models", exist_ok=True)
            fold_model_path = f"cv_models/best_model_fold_{fold+1}.pt"

        best_state, best_val_loss = train_fold(
            train_loader, val_loader, model, criterion, optimizer, scheduler, scaler,
            args.epochs, args.accumulation_steps, device, fold_model_path
        )

        # Evaluate on validation set (the fold)
        model.load_state_dict(best_state)
        model.to(device)
        val_labels_all, val_preds_all = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_labels_all, val_preds_all)
        val_f1 = f1_score(val_labels_all, val_preds_all, average='binary')
        fold_val_metrics.append({'accuracy': val_acc, 'f1': val_f1})
        print(f"Fold {fold+1} Val | Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # If test set exists, evaluate on it
        if len(X_test) > 0:
            test_loader = DataLoader(
                BinaryDataset(X_test, y_test),
                batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
            )
            test_labels_all, test_preds_all = evaluate(model, test_loader, device)
            test_acc = accuracy_score(test_labels_all, test_preds_all)
            test_f1 = f1_score(test_labels_all, test_preds_all, average='binary')
            fold_test_metrics.append({'accuracy': test_acc, 'f1': test_f1})
            print(f"Fold {fold+1} Test | Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

        best_models.append(best_state)

    # Summary
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    avg_val_acc = np.mean([m['accuracy'] for m in fold_val_metrics])
    std_val_acc = np.std([m['accuracy'] for m in fold_val_metrics])
    avg_val_f1 = np.mean([m['f1'] for m in fold_val_metrics])
    std_val_f1 = np.std([m['f1'] for m in fold_val_metrics])

    print(f"Validation (on folds):")
    print(f"  Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"  F1 (binary): {avg_val_f1:.4f} ± {std_val_f1:.4f}")

    if len(X_test) > 0:
        avg_test_acc = np.mean([m['accuracy'] for m in fold_test_metrics])
        std_test_acc = np.std([m['accuracy'] for m in fold_test_metrics])
        avg_test_f1 = np.mean([m['f1'] for m in fold_test_metrics])
        std_test_f1 = np.std([m['f1'] for m in fold_test_metrics])
        print(f"\nTest set (held-out):")
        print(f"  Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"  F1 (binary): {avg_test_f1:.4f} ± {std_test_f1:.4f}")

        # Train final model on all train data and evaluate on test
        print("\nTraining final model on all train data...")
        final_model = TrainableBinaryClassifier(args.encoder_name, args.max_seq_length, num_classes=2).to(device)
        final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr)
        final_scheduler = torch.optim.lr_scheduler.LinearLR(final_optimizer, start_factor=1.0, end_factor=0.1, total_iters=3)
        final_scaler = GradScaler()
        final_loader = DataLoader(
            BinaryDataset(X_train, y_train),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        # Train final model
        for epoch in range(args.epochs):
            final_model.train()
            total_loss = 0
            final_optimizer.zero_grad()
            for i, (texts_b, labels) in enumerate(final_loader):
                texts_b = list(texts_b)
                labels = labels.to(device)
                with autocast():
                    logits = final_model(texts_b)
                    loss = criterion(logits, labels)
                    loss = loss / args.accumulation_steps
                final_scaler.scale(loss).backward()
                if (i + 1) % args.accumulation_steps == 0:
                    final_scaler.step(final_optimizer)
                    final_scaler.update()
                    final_optimizer.zero_grad()
                total_loss += loss.item() * args.accumulation_steps
            final_scheduler.step()
            # No validation, just train
            print(f"  Epoch {epoch+1} | Train Loss: {total_loss / len(final_loader):.4f}")

        # Evaluate final model on test set
        test_loader = DataLoader(
            BinaryDataset(X_test, y_test),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        final_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts_b, labels in test_loader:
                texts_b = list(texts_b)
                labels = labels.to(device)
                logits = final_model(texts_b)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        final_test_acc = accuracy_score(all_labels, all_preds)
        final_test_f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"\nFinal model on test set:")
        print(f"  Accuracy: {final_test_acc:.4f}")
        print(f"  F1 (binary): {final_test_f1:.4f}")
        # Save final model
        torch.save(final_model.state_dict(), args.best_model_path)
        print(f"Final model saved to {args.best_model_path}")

    if args.save_fold_models:
        print(f"\nFold models saved in cv_models/ (best per fold).")
    else:
        print("\nNo fold models saved (use -save_fold_models to save each fold's best).")

    # Save aggregated results to a file
    with open("cv_results.json", "w") as f:
        json.dump({
            "fold_val_metrics": fold_val_metrics,
            "fold_test_metrics": fold_test_metrics if len(X_test) > 0 else None,
            "summary": {
                "val_accuracy_mean": avg_val_acc,
                "val_accuracy_std": std_val_acc,
                "val_f1_mean": avg_val_f1,
                "val_f1_std": std_val_f1,
                "test_accuracy_mean": avg_test_acc if len(X_test) > 0 else None,
                "test_accuracy_std": std_test_acc if len(X_test) > 0 else None,
                "test_f1_mean": avg_test_f1 if len(X_test) > 0 else None,
                "test_f1_std": std_test_f1 if len(X_test) > 0 else None,
                "final_test_accuracy": final_test_acc if len(X_test) > 0 else None,
                "final_test_f1": final_test_f1 if len(X_test) > 0 else None,
            }
        }, f, indent=2)

    print("\nResults saved to cv_results.json")


if __name__ == "__main__":
    main()