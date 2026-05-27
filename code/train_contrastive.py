#!/usr/bin/env python3
import json
import argparse
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contrastively fine-tune a sentence encoder on (fallacious, neutral) pairs."
    )
    parser.add_argument('-pairs_file', type=str, default='./all_pairs.json',
                        help='JSON list of {original_text, neutral_text, original_label}.')
    parser.add_argument('-model_name', type=str, default='deepvk/USER-bge-m3',
                        help='Base sentence-transformer encoder.')
    parser.add_argument('-output_dir', type=str, default='user-bge-contrastive',
                        help='Trainer output/checkpoint directory.')
    parser.add_argument('-save_dir', type=str, default='user-bge-contrastive-finetuned',
                        help='Where to save the final fine-tuned encoder.')
    parser.add_argument('-max_seq_length', type=int, default=512)
    parser.add_argument('-num_train_epochs', type=int, default=5)
    parser.add_argument('-per_device_train_batch_size', type=int, default=2)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('-warmup_ratio', type=float, default=0.1)
    parser.add_argument('-scale', type=float, default=20.0,
                        help='Scale for CachedMultipleNegativesRankingLoss.')
    parser.add_argument('-eval_steps', type=int, default=200)
    parser.add_argument('-logging_steps', type=int, default=50)
    parser.add_argument('-test_size', type=float, default=0.1,
                        help='Validation fraction for the pair split.')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-no_fp16', action='store_true', help='Disable FP16 training.')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.pairs_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences1 = [item["original_text"] for item in data]
    sentences2 = [item["neutral_text"] for item in data]

    labels_for_split = [item["original_label"] for item in data]
    train_idx, val_idx = train_test_split(
        range(len(sentences1)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels_for_split
    )

    train_s1 = [sentences1[i] for i in train_idx]
    train_s2 = [sentences2[i] for i in train_idx]
    val_s1 = [sentences1[i] for i in val_idx]
    val_s2 = [sentences2[i] for i in val_idx]

    train_dataset = Dataset.from_dict({"sentence1": train_s1, "sentence2": train_s2})
    val_dataset = Dataset.from_dict({"sentence1": val_s1, "sentence2": val_s2})

    print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    train_loss = CachedMultipleNegativesRankingLoss(model, scale=args.scale)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=not args.no_fp16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
    )

    trainer.train()

    model.save_pretrained(args.save_dir)
    print(f"Contrastive training complete, model saved to {args.save_dir}.")

    print("\nEvaluating quality on validation pairs (Logistic Regression on embeddings):")
    val_texts = []
    val_binary = []
    for i in range(len(val_s1)):
        val_texts.append(val_s1[i]); val_binary.append(1)
        val_texts.append(val_s2[i]); val_binary.append(0)

    embeddings = model.encode(val_texts, show_progress_bar=True)
    X_train_emb, X_val_emb, y_train_emb, y_val_emb = train_test_split(
        embeddings, val_binary, test_size=0.5, random_state=args.seed
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_emb, y_train_emb)
    acc = accuracy_score(y_val_emb, clf.predict(X_val_emb))
    print(f"LogReg accuracy on val pairs (quick check): {acc:.4f}")


if __name__ == "__main__":
    main()