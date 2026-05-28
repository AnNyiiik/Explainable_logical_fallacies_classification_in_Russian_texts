#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for TACEI (two-phase MTL + CLS) with gradient accumulation.
Supports large models and long sequences by using gradient accumulation and mixed precision.
"""
import torch
import argparse
import numpy as np
import pandas as pd
import optuna
import random
from types import SimpleNamespace

import utils
from mtlPredictor import MTLTrainer
from classPredictor import CLSTrainer

random.seed(12345)
np.random.seed(67891)
torch.manual_seed(54321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import wandb
wandb.init(mode="disabled")

LABEL_IDX_MAP = {
    'ad hominem': 0,
    'ad populum': 1,
    'appeal to emotion': 2,
    'circular reasoning': 3,
    'equivocation': 4,
    'fallacy of credibility': 5,
    'fallacy of extension': 6,
    'fallacy of relevance': 7,
    'false causality': 8,
    'false dilemma': 9,
    'faulty generalization': 10,
    'intentional': 11,
}
NUM_CLASSES = len(LABEL_IDX_MAP)
SEP_EXP_TOKEN = " _sep_exp_token_ "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pre_split_data(train_file, valid_file, test_file):
    """Load already split train/valid/test files and apply same preprocessing as in main.py"""
    prepro_exp = 'prepro_exp'
    prepro_text = 'prepro_text'
    prepro_label = 'prepro_label'

    def preprocess_df(df):
        df[prepro_exp] = df['rationale'].apply(lambda x: x.strip().replace('[SEP]', SEP_EXP_TOKEN))
        df[prepro_exp] = df[prepro_exp].apply(lambda x: utils.preprocess_text(x, lower=True))
        df[prepro_exp] = df[prepro_exp].apply(lambda x: [y.strip() for y in x.split(SEP_EXP_TOKEN)])
        df[prepro_text] = df['text'].apply(lambda x: utils.preprocess_text(x, lower=True))
        df[prepro_label] = df['label'].apply(lambda x: LABEL_IDX_MAP[x])
        df.drop_duplicates(subset=prepro_text, inplace=True)
        return df

    train_df = pd.read_csv(train_file, delimiter='\t')
    valid_df = pd.read_csv(valid_file, delimiter='\t')
    test_df = pd.read_csv(test_file, delimiter='\t')

    train_df = preprocess_df(train_df)
    valid_df = preprocess_df(valid_df)
    test_df = preprocess_df(test_df)

    train_text = train_df[prepro_text].values
    train_cls = train_df[prepro_label].values
    train_exp = train_df[prepro_exp].values

    valid_text = valid_df[prepro_text].values
    valid_cls = valid_df[prepro_label].values
    valid_exp = valid_df[prepro_exp].values

    test_text = test_df[prepro_text].values
    test_cls = test_df[prepro_label].values
    test_exp = test_df[prepro_exp].values

    print(f"Loaded splits: train={len(train_text)}, valid={len(valid_text)}, test={len(test_text)}")
    return (train_text, train_cls, train_exp,
            valid_text, valid_cls, valid_exp,
            test_text, test_cls, test_exp)

def objective(trial, data_tuple, fixed_args):
    """
    Objective function for Optuna.
    Tunes hyperparameters and returns test macro F1.
    """
    effective_batch_size = trial.suggest_categorical('batch_size', [8, 16])
    lr = trial.suggest_float('lr', 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.1, log=True)
    n_epochs = trial.suggest_int('n_epochs', 5, 15, step=5)
    patience = trial.suggest_int('patience', 3, 10, step=1)
    exp_weight = trial.suggest_float('exp_weight', 0.2, 1.0, step=0.05)

    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float)
    w_ap = trial.suggest_float('w_ad_populum', 1.0, 3.0, step=0.2)
    w_cr = trial.suggest_float('w_circular_reasoning', 1.0, 3.0, step=0.2)
    w_int = trial.suggest_float('w_intentional', 1.0, 3.0, step=0.2)
    w_fc = trial.suggest_float('w_false_causality', 1.0, 2.5, step=0.2)
    w_fcrd = trial.suggest_float('w_fallacy_credibility', 0.8, 1.5, step=0.1)
    class_weights[1] = w_ap   # ad populum
    class_weights[3] = w_cr   # circular reasoning
    class_weights[11] = w_int # intentional
    class_weights[8] = w_fc   # false causality
    class_weights[5] = w_fcrd # fallacy of credibility

    physical_batch_size = 4
    accumulation_steps = effective_batch_size // physical_batch_size
    if effective_batch_size % physical_batch_size != 0:
        effective_batch_size = physical_batch_size * (effective_batch_size // physical_batch_size)
        accumulation_steps = effective_batch_size // physical_batch_size
        print(f"Adjusted effective batch size to {effective_batch_size} for divisibility")

    args = SimpleNamespace(
        model_config=fixed_args.model_config,
        lr=lr,
        weight_decay=weight_decay,
        exp_weight=exp_weight,
        train_batch_size=physical_batch_size,
        test_batch_size=physical_batch_size,
        accumulation_steps=accumulation_steps,
        n_epochs=n_epochs,
        patience=patience,
        cls_hidden_size=fixed_args.cls_hidden_size,
        exp_hidden_size=fixed_args.exp_hidden_size,
        max_len=fixed_args.max_len,
        full_train=False,
        device=DEVICE,
        saved_model_path=None,
        label_idx_map=LABEL_IDX_MAP,
        class_weights=class_weights,
        use_amp=fixed_args.use_amp,
    )

    (train_text, train_cls, train_exp,
     valid_text, valid_cls, valid_exp,
     test_text, test_cls, test_exp) = data_tuple

    print(f"\nTrial {trial.number}: eff_bs={effective_batch_size}, phys_bs={physical_batch_size}, "
          f"acc_steps={accumulation_steps}, max_len={args.max_len}, use_amp={args.use_amp}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        print(f"Phase1 | lr={lr:.2e}, wd={weight_decay:.2e}, epochs={n_epochs}, pat={patience}, exp_w={exp_weight:.2f}")
        mtl_trainer = MTLTrainer(args, train_text, train_cls, train_exp)
        (train_exp_pred, train_labels,
         valid_exp_pred, valid_labels,
         test_exp_pred, test_labels) = mtl_trainer.eval(
            train_indices=np.arange(len(train_text)),
            valid_indices=np.arange(len(valid_text)),
            test_indices=np.arange(len(test_text))
        )

        print(f"Phase2 | lr={lr:.2e}, wd={weight_decay:.2e}, epochs={n_epochs}, pat={patience}")
        cls_trainer = CLSTrainer(args, train_exp_pred, train_labels)
        test_f1, _, _ = cls_trainer.eval(
            train_data=train_exp_pred, train_labels=train_labels,
            valid_data=valid_exp_pred, valid_labels=valid_labels,
            test_data=test_exp_pred, test_labels=test_labels
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"Trial {trial.number} failed due to OOM: {e}. Returning -1.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return -1.0
    except Exception as e:
        print(f"Trial {trial.number} failed with unexpected error: {e}")
        return -1.0

    trial.report(test_f1, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return test_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuning for TACEI with gradient accumulation")
    parser.add_argument('-train_file', type=str, required=True, help='Path to train TSV file')
    parser.add_argument('-valid_file', type=str, required=True, help='Path to validation TSV file')
    parser.add_argument('-test_file', type=str, required=True, help='Path to test TSV file')
    parser.add_argument('-model_config', type=str, default='deepvk/USER-bge-m3', help='HuggingFace model name')
    parser.add_argument('-max_len', type=int, default=256, help='Maximum sequence length (tokens)')
    parser.add_argument('-cls_hidden_size', type=int, default=128, help='Hidden size for classification head')
    parser.add_argument('-exp_hidden_size', type=int, default=128, help='Hidden size for explanation GRU')
    parser.add_argument('-n_trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('-study_name', type=str, default='tacei_optuna', help='Study name for storage')
    parser.add_argument('-storage', type=str, default=None, help='SQLite URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('-use_amp', action='store_true', help='Enable automatic mixed precision (FP16)')
    args_opt = parser.parse_args()

    data = load_pre_split_data(args_opt.train_file, args_opt.valid_file, args_opt.test_file)

    study = optuna.create_study(
        study_name=args_opt.study_name,
        storage=args_opt.storage,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(trial, data, args_opt),
        n_trials=args_opt.n_trials,
        show_progress_bar=True
    )

    print("\n" + "="*60)
    print("Best trial results:")
    best_trial = study.best_trial
    print(f"  Test macro F1: {best_trial.value:.4f}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_weights = torch.ones(NUM_CLASSES)
    best_weights[1] = best_trial.params.get('w_ad_populum', 1.0)
    best_weights[3] = best_trial.params.get('w_circular_reasoning', 1.0)
    best_weights[11] = best_trial.params.get('w_intentional', 1.0)
    best_weights[8] = best_trial.params.get('w_false_causality', 1.0)
    best_weights[5] = best_trial.params.get('w_fallacy_credibility', 1.0)
    print("\nBest class weights tensor (order as in LABEL_IDX_MAP):")
    print(best_weights.tolist())

    if args_opt.storage is None:
        import joblib
        joblib.dump(study, "optuna_study.pkl")
        print("Study saved to optuna_study.pkl")
    else:
        print(f"Study saved to storage: {args_opt.storage}")