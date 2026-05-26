#!/usr/bin/env python3
import os
import torch
import argparse
import random
import numpy as np
import pandas as pd
import utils
import wandb
import transformers.file_utils as hf_file_utils
from mtlPredictor import MTLTrainer
from classPredictor import CLSTrainer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
from transformers import logging

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")
import time

random.seed(12345)
np.random.seed(67891)
torch.manual_seed(54321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Default per-class weights (order must match label_idx_map below).
# Used as a fallback when -class_weights is not passed on the command line.
DEFAULT_CLASS_WEIGHTS = [
    1.0,  # ad hominem
    1.5,  # ad populum
    1.0,  # appeal to emotion
    2.0,  # circular reasoning
    1.0,  # equivocation
    1.2,  # fallacy of credibility
    0.8,  # fallacy of extension
    0.9,  # fallacy of relevance
    1.5,  # false causality
    1.0,  # false dilemma
    1.5,  # faulty generalization
    2.0,  # intentional
]


def patch_hf_relative_redirects():
    if not hasattr(hf_file_utils, "http_get"):
        return

    original_http_get = hf_file_utils.http_get

    def patched_http_get(url, temp_file, proxies=None, resume_size=0, headers=None):
        if isinstance(url, str) and url.startswith('/'):
            url = 'https://huggingface.co' + url
        return original_http_get(url, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

    hf_file_utils.http_get = patched_http_get


if __name__ == "__main__":
    patch_hf_relative_redirects()
    parser = argparse.ArgumentParser()
    begin_time = time.time()

    parser.add_argument('-input_path', type=str, default='data/train.tsv')
    parser.add_argument('-random_state', type=int, default=13)
    parser.add_argument('-model_config', type=str, default='vinai/bertweet-base')
    parser.add_argument('-cls_hidden_size', type=int, default=128)
    parser.add_argument('-exp_hidden_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('-exp_weight', type=float, default=0.07)
    parser.add_argument('-max_len', type=int, default=128)
    parser.add_argument('-n_epochs', type=int, default=20)
    parser.add_argument('-n_folds', type=int, default=5)
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('-test_size', type=float, default=15)
    parser.add_argument('-train_batch_size', type=int, default=16)
    parser.add_argument('-test_batch_size', type=int, default=64)
    parser.add_argument('-sep_exp_token', type=str, default=' _sep_exp_token_ ')
    parser.add_argument('-text_col', type=str, default='text')
    parser.add_argument('-label_col', type=str, default='label')
    parser.add_argument('-exp_col', type=str, default='rationale')
    parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-input_new_data_path', type=str, default="data/input.csv")
    parser.add_argument('-output_new_data_path', type=str, default="data/output.csv")
    parser.add_argument('-saved_model_path', type=str, default="data/saved_models/")
    parser.add_argument('-wandb_api_key', type=str, default=None, help='Weights & Biases API key')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='weight decay for AdamW')

    parser.add_argument('-class_weights', type=float, nargs='+', default=None,
                        help='Per-class weights as a space-separated list, ordered as in label_idx_map. '
                             'If omitted, DEFAULT_CLASS_WEIGHTS is used.')

    parser.add_argument('-full_train', type=str, default=True,
                        help='True: full data, False: only use the correct prediction from 1st phase for training in 2nd phase')

    parser.add_argument('-mode', type=str, default='eval', help='train: train model, eval:evaluate with n-folds, \
                                prediction: make prediction on new data')

    parser.add_argument('-train_file', type=str, default=None, help='Path to pre-split train file')
    parser.add_argument('-valid_file', type=str, default=None, help='Path to pre-split validation file')
    parser.add_argument('-test_file', type=str, default=None, help='Path to pre-split test file')

    label_idx_map = {
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

    prepro_exp = 'prepro_exp'
    prepro_text = 'prepro_text'
    prepro_label = 'prepro_label'

    args = parser.parse_args()

    # Build the class-weights tensor from the CLI argument, falling back to defaults.
    weights_list = args.class_weights if args.class_weights is not None else DEFAULT_CLASS_WEIGHTS
    if len(weights_list) != len(label_idx_map):
        parser.error(
            f"-class_weights expects {len(label_idx_map)} values (one per class), "
            f"got {len(weights_list)}."
        )
    args.class_weights = torch.tensor(weights_list, dtype=torch.float)
    print("Class weights:", args.class_weights.tolist(), flush=True)

    args.label_idx_map = label_idx_map

    idx_label_map = {idx: label for label, idx in label_idx_map.items()}

    wandb.init(
        project="TACEI_experiment",
        config={
            "model_config": args.model_config,
            "learning_rate": args.lr,
            "exp_weight": args.exp_weight,
            "train_batch_size": args.train_batch_size,
            "n_epochs": args.n_epochs,
            "patience": args.patience,
            "cls_hidden_size": args.cls_hidden_size,
            "exp_hidden_size": args.exp_hidden_size,
            "class_weights": args.class_weights.tolist(),
        }
    )

    if args.mode == 'prediction':
        mtlTrainer = MTLTrainer(args)
        clsTrainer = CLSTrainer(args)
        num_classes = len(label_idx_map)
        print("Load models...", flush=True)
        mtlTrainer.load(num_classes=num_classes, saved_model_path=args.saved_model_path + "phase1.pt")
        clsTrainer.load(num_classes=num_classes, saved_model_path=args.saved_model_path + "phase2.pt")

        print("Read new data...", flush=True)
        new_data = []
        prepro_data = []
        with open(args.input_new_data_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                new_data.append(line.strip())
                prepro_data.append(utils.preprocess_text(line.strip()))

        cls_pred_p1, _, exp_pred_data, exp_pred_masked, exp_pred_probs = mtlTrainer.classify(prepro_data)
        cls_pred_p2, cls_pred_probs = clsTrainer.classify(exp_pred_masked)

        output_dir = os.path.dirname(args.output_new_data_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        n = min(
            len(new_data),
            len(prepro_data),
            len(cls_pred_p1),
            len(exp_pred_data),
            len(exp_pred_masked),
            len(exp_pred_probs),
            len(cls_pred_p2),
            len(cls_pred_probs),
        )
        if n < len(new_data):
            print("Warning: output length mismatch detected. Saving first {} rows.".format(n), flush=True)

        cls_pred_p1_idx = [int(x) for x in cls_pred_p1[:n]]
        cls_pred_p2_idx = [int(x) for x in cls_pred_p2[:n]]
        cls_prob_p2 = [round(float(x), 4) for x in cls_pred_probs[:n]]

        output_new = pd.DataFrame({
            'text': new_data[:n],
            'preprocessed_text': prepro_data[:n],
            'phase1_label_idx': cls_pred_p1_idx,
            'phase1_label': [idx_label_map.get(i, 'unknown') for i in cls_pred_p1_idx],
            'phase1_rationale': exp_pred_data[:n],
            'phase1_masked_text': exp_pred_masked[:n],
            'phase1_token_probs': exp_pred_probs[:n],
            'phase2_label_idx': cls_pred_p2_idx,
            'phase2_label': [idx_label_map.get(i, 'unknown') for i in cls_pred_p2_idx],
            'phase2_confidence': cls_prob_p2,
        })
        output_new.to_csv(args.output_new_data_path, index=False)
        print("Saved prediction output to {}".format(args.output_new_data_path))

        print("Total time: ", time.time() - begin_time, flush=True)

    else:
        if args.mode == 'train' and args.train_file and args.valid_file and args.test_file:
            print("Loading pre-split datasets...")

            train_data = pd.read_csv(args.train_file, delimiter='\t')
            valid_data = pd.read_csv(args.valid_file, delimiter='\t')
            test_data = pd.read_csv(args.test_file, delimiter='\t')

            for df in [train_data, valid_data, test_data]:
                df[prepro_exp] = df[args.exp_col].apply(lambda x: x.strip().replace('[SEP]', args.sep_exp_token))
                df[prepro_exp] = df[prepro_exp].apply(lambda x: utils.preprocess_text(x, lower=True))
                df[prepro_exp] = df[prepro_exp].apply(lambda x: [y.strip() for y in x.split(args.sep_exp_token)])
                df[prepro_text] = df[args.text_col].apply(lambda x: utils.preprocess_text(x, lower=True))
                df[prepro_label] = df[args.label_col].apply(lambda x: label_idx_map[x])

            train_text = np.array(train_data[prepro_text])
            train_exp = np.array(train_data[prepro_exp])
            train_cls = np.array(train_data[prepro_label])

            valid_text = np.array(valid_data[prepro_text])
            valid_exp = np.array(valid_data[prepro_exp])
            valid_cls = np.array(valid_data[prepro_label])

            test_text = np.array(test_data[prepro_text])
            test_exp = np.array(test_data[prepro_exp])
            test_cls = np.array(test_data[prepro_label])

            print(f"Train size: {len(train_text)}, Valid size: {len(valid_text)}, Test size: {len(test_text)}")

            all_text = np.concatenate([train_text, valid_text, test_text])
            all_cls = np.concatenate([train_cls, valid_cls, test_cls])
            all_exp = np.concatenate([train_exp, valid_exp, test_exp])

            train_indices = np.arange(len(train_text))
            valid_indices = np.arange(len(train_text), len(train_text) + len(valid_text))
            test_indices = np.arange(len(train_text) + len(valid_text), len(all_text))

            mtlTrainer = MTLTrainer(args, all_text, all_cls, all_exp)
            print(">>>>> Phase 1...........", flush=True)
            train_exp_pred_data, train_labels, valid_exp_pred_data, valid_labels, test_exp_pred_data, test_labels = mtlTrainer.eval(
                train_indices, valid_indices, test_indices
            )

            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(mtlTrainer.model.state_dict(), args.saved_model_path + "phase1.pt")
            print(f"Phase 1 weights saved to {args.saved_model_path}phase1.pt")

            clsTrainer = CLSTrainer(args, train_exp_pred_data, train_labels)
            print(">>>>> Phase 2...........", flush=True)

            clsTrainer.eval(
                train_data=train_exp_pred_data,
                train_labels=train_labels,
                valid_data=valid_exp_pred_data,
                valid_labels=valid_labels,
                test_data=test_exp_pred_data,
                test_labels=test_labels
            )

            torch.save(clsTrainer.model.state_dict(), args.saved_model_path + "phase2.pt")
            print(f"Phase 2 weights saved to {args.saved_model_path}phase2.pt")

            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print(f"📁 Models saved to: {args.saved_model_path}")
            print("   - phase1.pt (Phase 1 weights)")
            print("   - phase2.pt (Phase 2 weights)")
            print("   - phase1_best_checkpoint.pt (full Phase 1 checkpoint)")
            print("   - phase2_best_checkpoint.pt (full Phase 2 checkpoint)")
            print("=" * 60)

        else:
            # load data
            data = pd.read_csv(args.input_path, delimiter='\t')

            # preprocess exp
            data[prepro_exp] = data[args.exp_col].apply(lambda x: x.strip().replace('[SEP]', args.sep_exp_token))
            data[prepro_exp] = data[prepro_exp].apply(lambda x: utils.preprocess_text(x, lower=True))
            data[prepro_exp] = data[prepro_exp].apply(lambda x: [y.strip() for y in x.split(args.sep_exp_token)])
            # preprocess text
            data[prepro_text] = data[args.text_col].apply(lambda x: utils.preprocess_text(x, lower=True))

            data[prepro_label] = data[args.label_col].apply(lambda x: label_idx_map[x])

            data.drop_duplicates(subset=prepro_text, inplace=True)
            print("Data: ", data.shape, flush=True)

            # initialize models
            text_data = np.array(data[prepro_text])
            exp_data = np.array(data[prepro_exp])
            cls_data = np.array(data[prepro_label])

            if args.mode == 'train':
                os.makedirs(args.saved_model_path, exist_ok=True)
                # train models on entire data
                mtlTrainer = MTLTrainer(args, text_data, cls_data, exp_data)
                print(">>>>> Phase 1...........", flush=True)
                train_exp_data, train_labels = mtlTrainer.train(saved_model_path=args.saved_model_path + "phase1.pt")
                print(">>>>> Phase 2...........", flush=True)
                clsTrainer = CLSTrainer(args, train_exp_data, cls_data)
                clsTrainer.train(saved_model_path=args.saved_model_path + "phase2.pt")

            elif args.mode == "eval":
                mtlTrainer = MTLTrainer(args, text_data, cls_data, exp_data)
                clsTrainer = CLSTrainer(args, text_data, cls_data)
                # cross-validate
                # Используем ShuffleSplit без стратификации для второго разбиения
                kfold = StratifiedShuffleSplit(n_splits=args.n_folds, test_size=args.test_size / 100,
                                               random_state=args.random_state)
                fold = 0
                for train_indices, remain_indices in kfold.split(text_data, cls_data):
                    # Для второго разбиения используем обычный train_test_split без стратификации
                    valid_indices, test_indices = train_test_split(remain_indices, test_size=0.5,
                                                                   random_state=args.random_state)
                    print("---------------------FOLD {}-----------------------".format(fold))
                    print(">>>>> Phase 1...........", flush=True)
                    train_exp_pred_data, train_labels, valid_exp_pred_data, \
                        valid_labels, test_exp_pred_data, test_labels = mtlTrainer.eval(train_indices, valid_indices,
                                                                                        test_indices)

                    print(">>>>> Phase 2............", flush=True)
                    clsTrainer.eval(train_exp_pred_data, train_labels, valid_exp_pred_data, valid_labels,
                                    test_exp_pred_data, test_labels)
                    print("--------------------------------------------------", flush=True)
                    fold += 1