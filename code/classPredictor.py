import numpy as np
import transformers
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import OrderedDict
import torch
import torch.nn as nn
import random
import utils
import time
import sys
import os
import wandb
from utils import per_class_classification_analysis
from contextlib import nullcontext


class CLSModel(nn.Module):
    def __init__(self, bert_config="vinai/bertweet-base", hidden_size=768, num_classes=6):
        super(CLSModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes, bias=True)
        self.num_classes = num_classes

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, return_dict=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=return_dict)
        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
            pooled_output = getattr(outputs, "pooler_output", None)
        else:
            sequence_output = outputs[0]
            pooled_output = outputs[1] if len(outputs) > 1 else None

        if pooled_output is None:
            pooled_output = sequence_output[:, 0]

        outputs = self.linear(self.dropout(pooled_output))
        return outputs


def save_full_checkpoint_phase2(model, optimizer, scheduler, epoch, best_loss, args, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'args': args,
        'num_classes': model.num_classes if hasattr(model, 'num_classes') else None,
        'class_weights': args.class_weights.cpu() if hasattr(args, 'class_weights') and args.class_weights is not None else None,
        'label_idx_map': args.label_idx_map if hasattr(args, 'label_idx_map') else None,
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Full Phase 2 checkpoint saved to {filepath}")


def load_full_checkpoint_phase2(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"✅ Phase 2 checkpoint loaded from {filepath} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


class CLSTrainer:
    def __init__(self, args, data=None, labels=None):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_config, use_fast=False)
        if data is not None:
            self.data = np.array(data)
            self.labels = np.array(labels)
            self.num_classes = int(np.max(self.labels)) + 1

    def predict(self, data, labels=None, loss_func=None, batch_size=128):
        self.model.eval()
        total_loss = 0
        outputs = []
        output_probs = []
        batch_num = 0
        with torch.no_grad():
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                X_batch = data[batch_start:batch_end]
                _, sents_tensor, masks_tensor, _ = utils.tokenize_text(
                    self.tokenizer, X_batch, self.tokenizer.pad_token, max_len=self.args.max_len
                )
                sents_tensor = torch.tensor(sents_tensor, dtype=torch.long).to(self.args.device)
                masks_tensor = torch.tensor(masks_tensor, dtype=torch.long).to(self.args.device)
                out = self.model(input_ids=sents_tensor, attention_mask=masks_tensor)
                if labels is not None:
                    y_batch = labels[batch_start:batch_end]
                    loss = loss_func(out, torch.tensor(y_batch, dtype=torch.long).to(self.args.device))
                    total_loss += loss.item()
                out = nn.Softmax(dim=-1)(out).max(dim=-1)
                outputs += out.indices.cpu()
                output_probs += out.values.cpu()
                batch_num += 1
        loss = total_loss / batch_num if batch_num > 0 else 0
        return loss, outputs, output_probs

    def predict_proba(self, data, batch_size=128):
        self.model.eval()
        all_probs = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                X_batch = data[batch_start:batch_end]
                _, sents_tensor, masks_tensor, _ = utils.tokenize_text(
                    self.tokenizer, X_batch, self.tokenizer.pad_token, max_len=self.args.max_len
                )
                sents_tensor = torch.tensor(sents_tensor, dtype=torch.long).to(device)
                masks_tensor = torch.tensor(masks_tensor, dtype=torch.long).to(device)
                out = self.model(input_ids=sents_tensor, attention_mask=masks_tensor)
                probs = torch.nn.Softmax(dim=-1)(out).cpu().numpy()
                all_probs.extend(probs)
        return all_probs

    def fit(self, data, labels, batch_size, loss_func, accumulation_steps=1):
        self.model.train()
        train_loss = 0
        epoch_indices = random.sample(range(len(data)), len(data))
        batch_num = 0

        use_amp = getattr(self.args, 'use_amp', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.optimizer.zero_grad()

        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch_indices = epoch_indices[batch_start:batch_end]
            X_batch = data[batch_indices]
            y_batch = labels[batch_indices]
            _, sents_tensor, masks_tensor, _ = utils.tokenize_text(
                self.tokenizer, X_batch, self.tokenizer.pad_token, max_len=self.args.max_len
            )
            sents_tensor = torch.tensor(sents_tensor, dtype=torch.long).to(self.args.device)
            masks_tensor = torch.tensor(masks_tensor, dtype=torch.long).to(self.args.device)

            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                output = self.model(input_ids=sents_tensor, attention_mask=masks_tensor)
                loss = loss_func(output, y_batch) / accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * accumulation_steps
            batch_num += 1

            if (batch_num % accumulation_steps == 0) or (batch_end == len(data)):
                if use_amp:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

        return train_loss / batch_num if batch_num > 0 else 0

    def eval(self, train_data=None, train_labels=None, valid_data=None, valid_labels=None,
             test_data=None, test_labels=None, train_indices=None, valid_indices=None, test_indices=None):
        if train_indices is not None:
            train_data = self.data[train_indices]
            valid_data = self.data[valid_indices]
            test_data = self.data[test_indices]
            train_labels = self.labels[train_indices]
            valid_labels = self.labels[valid_indices]
            test_labels = self.labels[test_indices]

        if train_data is None or len(train_data) == 0:
            print("Error: train_data is None or empty")
            return None, None, None

        train_data = np.array([self.tokenizer.cls_token + " " + str(x) + " " + self.tokenizer.sep_token for x in train_data])
        if valid_data is not None and len(valid_data) > 0:
            valid_data = np.array([self.tokenizer.cls_token + " " + str(x) + " " + self.tokenizer.sep_token for x in valid_data])
        else:
            valid_data = None
        if test_data is not None and len(test_data) > 0:
            test_data = np.array([self.tokenizer.cls_token + " " + str(x) + " " + self.tokenizer.sep_token for x in test_data])

        self.model = CLSModel(self.args.model_config, num_classes=self.num_classes)
        self.model.to(self.args.device)

        weight_decay = getattr(self.args, 'weight_decay', 0.01)
        self.optimizer = AdamW(self.model.parameters(), self.args.lr, eps=1e-8, weight_decay=weight_decay)

        n_batches = int(np.ceil(len(train_data) / self.args.train_batch_size))
        total_step = n_batches * self.args.n_epochs
        num_warmup_steps = int(0.1 * total_step)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=num_warmup_steps,
                                                                      num_training_steps=total_step)

        if hasattr(self.args, 'class_weights') and self.args.class_weights is not None:
            class_weights = self.args.class_weights.to(self.args.device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            print("Using class weights for Phase 2")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print("No class weights for Phase 2")

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(self.args.device)
        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)

        best_loss = float('inf')
        best_epoch = 0
        best_model_state_dict = None

        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_data, train_labels_tensor, self.args.train_batch_size, criterion, accumulation_steps)

            valid_loss = 0
            if valid_data is not None and valid_labels is not None:
                valid_loss, _, _ = self.predict(valid_data, valid_labels, criterion, self.args.test_batch_size)

            print("Epoch: %d, train_loss: %.3f, valid_loss: %.3f, time: %.3f" % (epoch, train_loss, valid_loss,
                                                                                 time.time() - begin_time), flush=True)
            wandb.log({
                "phase2_epoch": epoch,
                "phase2_train_loss": train_loss,
                "phase2_valid_loss": valid_loss,
                "phase2_learning_rate": self.scheduler.get_last_lr()[0]
            })

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in self.model.state_dict().items()})
                print(f"  ✅ New best model! valid_loss: {best_loss:.4f}")
                if hasattr(self.args, 'saved_model_path') and self.args.saved_model_path:
                    checkpoint_path = os.path.join(self.args.saved_model_path, f"phase2_best_checkpoint.pt")
                    save_full_checkpoint_phase2(self.model, self.optimizer, self.scheduler, epoch, best_loss, self.args, checkpoint_path)

            if epoch - best_epoch > self.args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print("+ Training ends!")
        print("+ Load best model from epoch {} with valid_loss: {:.4f}".format(best_epoch, best_loss))
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        self.model.to(self.args.device)

        if test_data is not None and test_labels is not None:
            _, y_pred, y_probs = self.predict(test_data, test_labels, criterion, batch_size=self.args.test_batch_size)
            test_f1 = f1_score(test_labels, y_pred, average='macro')
            print("++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("++ CLS F1 on test: %.4f" % test_f1)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++", flush=True)

            label_idx_to_name = {v: k for k, v in self.args.label_idx_map.items()} if hasattr(self.args, 'label_idx_map') else {}
            if label_idx_to_name:
                class_names = [label_idx_to_name[i] for i in range(self.num_classes) if i in label_idx_to_name]
            else:
                class_names = [str(i) for i in range(self.num_classes)]

            print("\n" + "🔍 " + "=" * 67)
            per_class_classification_analysis(y_true=test_labels, y_pred=y_pred, class_names=class_names,
                                              phase_name="Phase2_Test", logger=wandb.log)
            return test_f1, y_pred, y_probs

        self.model.cpu()
        self.optimizer = None
        self.scheduler = None
        if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
            torch.cuda.empty_cache()
        return None, None, None

    def train(self, data=None, labels=None, saved_model_path=None):
        if data is not None:
            self.data = np.array(data)
            self.labels = np.array(labels)
            self.num_classes = int(np.max(self.labels)) + 1

        train_data = np.array([self.tokenizer.cls_token + " " + x + " " + self.tokenizer.sep_token for x in self.data])
        self.model = CLSModel(self.args.model_config, num_classes=self.num_classes)

        weight_decay = getattr(self.args, 'weight_decay', 0.01)
        self.optimizer = AdamW(self.model.parameters(), self.args.lr, eps=1e-8, weight_decay=weight_decay)

        n_batches = int(np.ceil(len(train_data) / self.args.train_batch_size))
        total_step = n_batches * self.args.n_epochs
        num_warmup_steps = int(0.1 * total_step)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=num_warmup_steps,
                                                                      num_training_steps=total_step)

        self.model.to(self.args.device)

        if hasattr(self.args, 'class_weights') and self.args.class_weights is not None:
            class_weights = self.args.class_weights.to(self.args.device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            print("✅ Using class weights for Phase 2")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print("⚠️ No class weights for Phase 2")

        train_labels = torch.tensor(self.labels, dtype=torch.long).to(self.args.device)
        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)

        print("Training...", flush=True)
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_data, train_labels, self.args.train_batch_size, criterion, accumulation_steps)
            print("Epoch: %d, train_loss: %.3f, time: %.3f" % (epoch, train_loss, time.time() - begin_time), flush=True)
            wandb.log({
                "phase2_train_epoch": epoch,
                "phase2_train_loss": train_loss,
                "phase2_learning_rate": self.scheduler.get_last_lr()[0]
            })

        _, y_pred, _ = self.predict(self.data, self.labels, criterion, batch_size=self.args.test_batch_size)
        train_f1 = f1_score(self.labels, y_pred, average='macro')
        print("++ Train CLS F1: {}".format(train_f1), flush=True)
        wandb.log({"phase2_final_train_f1": train_f1})

        self.model.cpu()
        self.optimizer = None
        self.scheduler = None
        if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
            torch.cuda.empty_cache()
        if saved_model_path is not None:
            print("Save model to path: {}".format(saved_model_path), flush=True)
            torch.save(self.model.state_dict(), saved_model_path)
        return train_f1

    def load(self, num_classes=6, saved_model_path=None):
        if saved_model_path is None:
            print("Please enter the model path...")
            sys.exit(-1)
        try:
            self.model = CLSModel(self.args.model_config, num_classes=num_classes)
            self.model.load_state_dict(torch.load(saved_model_path))
            self.num_classes = num_classes
            print("✅ Model loaded from {}".format(saved_model_path))
        except Exception as e:
            print("Exception")
            print(e, flush=True)

    def classify(self, new_data):
        data = np.array([self.tokenizer.cls_token + " " + x + " " + self.tokenizer.sep_token for x in new_data])
        self.model.to(self.args.device)
        _, y_preds, y_probs = self.predict(data, batch_size=self.args.test_batch_size)
        self.model.cpu()
        if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
            torch.cuda.empty_cache()
        return y_preds, y_probs