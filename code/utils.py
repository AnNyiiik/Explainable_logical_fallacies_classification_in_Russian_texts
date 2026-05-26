import numpy as np
import re
import torch
from tweet_preprocessing import normalizeTweet
from sklearn.metrics import classification_report, confusion_matrix

def convert_sents_to_ids_tensor(tokenizer, sents, pad_token='<pad>'):
    tokens_list = [tokenizer.tokenize(sent) for sent in sents]
    sents_lengths = torch.tensor([len(tokens) for tokens in tokens_list])
    tokens_list_padded = pad_sents(tokens_list, pad_token)
    masks = np.asarray(tokens_list_padded) != pad_token
    masks_tensor = torch.tensor(masks, dtype=torch.long)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long)
    return sents_tensor, masks_tensor, sents_lengths

def create_only_rationale(original_text_tokens, importance_mask):
    if len(original_text_tokens) != len(importance_mask):
        raise ValueError("Length mismatch")
    only_rationale_tokens = [tok for tok, keep in zip(original_text_tokens, importance_mask) if keep == 1]
    return ' '.join(only_rationale_tokens)

def create_text_without_rationale(original_text_tokens, importance_mask):
    if len(original_text_tokens) != len(importance_mask):
        raise ValueError("Length mismatch")
    without_rationale_tokens = [tok if keep == 0 else '*' for tok, keep in zip(original_text_tokens, importance_mask)]
    return ' '.join(without_rationale_tokens)

def per_class_classification_analysis(y_true, y_pred, class_names=None, phase_name="", logger=None):
    if class_names is None:
        class_names = [str(i) for i in sorted(set(y_true) | set(y_pred))]
    print("\n" + "=" * 70)
    print(f"PER-CLASS CLASSIFICATION ANALYSIS - {phase_name}")
    print("=" * 70)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    print(f"\n{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for class_name in class_names:
        if class_name in report and isinstance(report[class_name], dict):
            m = report[class_name]
            print(f"{str(class_name):<30} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f} {m['support']:<10.0f}")
    print("-" * 80)
    print(f"{'macro avg':<30} {report['macro avg']['precision']:<12.4f} {report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
    print(f"{'weighted avg':<30} {report['weighted avg']['precision']:<12.4f} {report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
    cm = confusion_matrix(y_true, y_pred)
    errors = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i][j] > 0:
                errors.append((true_name, pred_name, cm[i][j]))
    errors.sort(key=lambda x: x[2], reverse=True)
    if errors:
        print("\n" + "=" * 70)
        print("TOP 5 MOST FREQUENT CONFUSIONS:")
        print(f"{'True Class':<30} {'Predicted as':<30} {'Count':<5}")
        print("-" * 70)
        for true_class, pred_class, count in errors[:5]:
            print(f"{true_class:<30} {pred_class:<30} {count:<5}")
    if logger is not None:
        for class_name in class_names:
            if class_name in report and isinstance(report[class_name], dict):
                safe_class_name = str(class_name).replace(' ', '_').replace('-', '_')
                logger({f"{phase_name}_precision_{safe_class_name}": report[class_name]['precision'],
                        f"{phase_name}_recall_{safe_class_name}": report[class_name]['recall'],
                        f"{phase_name}_f1_{safe_class_name}": report[class_name]['f1-score']})
        logger({f"{phase_name}_macro_precision": report['macro avg']['precision'],
                f"{phase_name}_macro_recall": report['macro avg']['recall'],
                f"{phase_name}_macro_f1": report['macro avg']['f1-score'],
                f"{phase_name}_accuracy": report.get('accuracy', 0)})
    return report

def per_class_rationale_analysis(true_rationales, pred_rationales, class_names=None, class_labels=None, phase_name="",
                                 logger=None, label_idx_to_name=None):
    if class_labels is None:
        all_true = []
        all_pred = []
        for t, p in zip(true_rationales, pred_rationales):
            all_true.extend(t)
            all_pred.extend(p)
        if len(all_true) == 0:
            print(f"Warning: No rationale tokens found for {phase_name}")
            return {'f1': 0, 'precision': 0, 'recall': 0}
        f1 = f1_score(all_true, all_pred, zero_division=0)
        prec = precision_score(all_true, all_pred, zero_division=0)
        rec = recall_score(all_true, all_pred, zero_division=0)
        print("\n" + "=" * 70)
        print(f"RATIONALE EXTRACTION ANALYSIS - {phase_name} (Overall)")
        print("=" * 70)
        print(f"  Token-F1: {f1:.4f}")
        print(f"  Token-Precision: {prec:.4f}")
        print(f"  Token-Recall: {rec:.4f}")
        if logger is not None:
            logger({f"{phase_name}_token_f1": f1, f"{phase_name}_token_precision": prec, f"{phase_name}_token_recall": rec})
        return {'f1': f1, 'precision': prec, 'recall': rec}
    if class_names is None:
        class_names = sorted(set(class_labels))
    if label_idx_to_name is None:
        label_idx_to_name = {i: str(i) for i in range(max(class_labels) + 1)}
    print("\n" + "=" * 70)
    print(f"RATIONALE EXTRACTION ANALYSIS - {phase_name} (Per Class)")
    print("=" * 70)
    print(f"\n{'Class':<30} {'Token-F1':<12} {'Token-Prec':<12} {'Token-Rec':<12} {'Support':<10}")
    print("-" * 80)
    results = {}
    for class_name in class_names:
        class_true = []
        class_pred = []
        class_count = 0
        for i, label_idx in enumerate(class_labels):
            label_name = label_idx_to_name.get(label_idx, str(label_idx))
            if label_name == class_name or str(label_idx) == class_name:
                if i < len(true_rationales) and i < len(pred_rationales):
                    class_true.extend(true_rationales[i])
                    class_pred.extend(pred_rationales[i])
                    class_count += 1
        if class_count > 0 and len(class_true) > 0:
            f1 = f1_score(class_true, class_pred, zero_division=0)
            prec = precision_score(class_true, class_pred, zero_division=0)
            rec = recall_score(class_true, class_pred, zero_division=0)
            results[class_name] = {'f1': f1, 'precision': prec, 'recall': rec, 'support': class_count}
            print(f"{str(class_name):<30} {f1:<12.4f} {prec:<12.4f} {rec:<12.4f} {class_count:<10}")
    print("-" * 80)
    if results:
        avg_f1 = np.mean([v['f1'] for v in results.values()])
        avg_prec = np.mean([v['precision'] for v in results.values()])
        avg_rec = np.mean([v['recall'] for v in results.values()])
    else:
        avg_f1 = avg_prec = avg_rec = 0.0
        print("No valid rationale data for any class")
    print(f"{'macro avg':<30} {avg_f1:<12.4f} {avg_prec:<12.4f} {avg_rec:<12.4f}")
    if logger is not None:
        for class_name, metrics in results.items():
            safe_class_name = str(class_name).replace(' ', '_').replace('-', '_')
            logger({f"{phase_name}_token_f1_{safe_class_name}": metrics['f1'],
                    f"{phase_name}_token_precision_{safe_class_name}": metrics['precision'],
                    f"{phase_name}_token_recall_{safe_class_name}": metrics['recall']})
        logger({f"{phase_name}_token_macro_f1": avg_f1,
                f"{phase_name}_token_macro_precision": avg_prec,
                f"{phase_name}_token_macro_recall": avg_rec})
    return results

def text_to_exp_labels(org_text, explan_text):
    labels = org_text
    exp = {x: len(x.split(' ')) for x in explan_text}
    exp = {k: v for k, v in sorted(exp.items(), key=lambda x: x[1], reverse=True)}
    try:
        for chunk, lenx in exp.items():
            labels = re.sub(re.escape(chunk), 'U ' * len(chunk.split()), labels)
        labels = re.sub('[^U ]', '0', labels)
        labels = re.sub('  ', ' ', labels).strip().split(" ")
        labels = [1 if i == 'U' else 0 for i in labels]
    except Exception as e:
        print("Exception", e, flush=True)
    return labels

def map_exp_labels(tokenizer, sents, exp_labels):
    tokenized_sents = [' '.join(tokenizer.tokenize(sent)) for sent in sents]
    tokenized_exps = [[' '.join(tokenizer.tokenize(exp)) for exp in exps] for exps in exp_labels]
    exp_labels = [text_to_exp_labels(sent, exp) for sent, exp in zip(tokenized_sents, tokenized_exps)]
    max_length = max([len(sent.split()) for sent in tokenized_sents])
    exp_labels = [label + [0] * (max_length - len(label)) for label in exp_labels]
    return exp_labels

def tokenize_text(tokenizer, sents, padding_token='<pad>', max_len=256):
    if isinstance(sents, np.ndarray):
        sents = sents.tolist()
    sents = [str(s) for s in sents]

    pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else padding_token
    tokens_list = []
    tokens_spans = []
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        tokens_list.append(tokens)
        spans = [(i, i+1) for i in range(len(tokens))]
        tokens_spans.append(spans)

    tokens_list_padded = []
    attention_masks = []
    for tokens in tokens_list:
        pad_len = max_len - len(tokens)
        padded_tokens = tokens + [pad_token] * pad_len
        mask = [1] * len(tokens) + [0] * pad_len
        tokens_list_padded.append(padded_tokens)
        attention_masks.append(mask)

    tokens_id_list_padded = [tokenizer.convert_tokens_to_ids(toks) for toks in tokens_list_padded]
    return tokens_list, tokens_id_list_padded, attention_masks, tokens_spans

def resampling_rebalanced_crossentropy(seq_reduction='none'):
    def loss(y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)
        min_len = min(y_pred.size(1), y_true.size(1))
        y_pred = y_pred[:, :min_len]
        y_true = y_true[:, :min_len]
        prior_pos = torch.mean(y_true, dim=-1, keepdims=True)
        prior_neg = torch.mean(1 - y_true, dim=-1, keepdim=True)
        eps = 1e-10
        prior_pos = torch.clamp(prior_pos, min=eps)
        prior_neg = torch.clamp(prior_neg, min=eps)
        weight = y_true / prior_pos + (1 - y_true) / prior_neg
        ret = -weight * (y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))
        if seq_reduction == 'mean':
            return torch.mean(ret, dim=-1)
        return ret
    return loss

def pad_sents(sents, pad_token):
    sents_padded = []
    max_len = max(len(s) for s in sents)
    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)
    return sents_padded

def max_pooling(y_values, data_slides, data, prob=False):
    pooled_values = []
    for y, y_slides, text_data in zip(y_values, data_slides, data):
        pooled_y = []
        for tup in y_slides:
            window = y[tup[0]:tup[1]]
            if len(window) == 0:
                pooled_y.append(0.0 if prob else 0)
                continue
            if prob:
                pooled_y.append(round(float(max(window)), 2))
            else:
                pooled_y.append(int(max(window)))
        pooled_values.append(pooled_y)
    return pooled_values

def preprocess_text(text, lower=True):
    text = normalizeTweet(text)
    if lower:
        text = text.lower()
    return text