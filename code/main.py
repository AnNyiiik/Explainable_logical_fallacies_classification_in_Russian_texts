import os 
import sys 
import torch 
import argparse
import random
import numpy as np
import pandas as pd
import utils
import re
from mtlPredictor import MTLTrainer 
from classPredictor import CLSTrainer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    begin_time = time.time()
    
    parser.add_argument('-input_path', type = str, default = '/home/nguyen/cv_dataset/epidemic_classifier/data/final_covid.tsv')
    parser.add_argument('-random_state', type = int, default = 13)
    parser.add_argument('-model_config', type = str, default = 'vinai/bertweet-base')
    parser.add_argument('-cls_hidden_size', type = int, default = 128)
    parser.add_argument('-exp_hidden_size', type = int, default = 128)
    parser.add_argument('-lr', type = float, default = 2e-5, help = 'learning rate')
    parser.add_argument('-exp_weight', type = float, default = 0.07)
    parser.add_argument('-max_len', type = int, default = 128)
    parser.add_argument('-n_epochs', type = int, default = 20)
    parser.add_argument('-n_folds', type = int, default = 5)
    parser.add_argument('-patience', type = int, default = 5)
    parser.add_argument('-test_size', type = int, default = 0.15)
    parser.add_argument('-train_batch_size', type = int, default = 16)
    parser.add_argument('-test_batch_size', type = int, default = 64)
    parser.add_argument('-sep_exp_token', type = str, default = ' _sep_exp_token_ ')
    parser.add_argument('-id_col', type = str, default = 'tweet_id')
    parser.add_argument('-text_col', type = str, default = 'tweet_text')
    parser.add_argument('-label_col', type = str, default = 'label')
    parser.add_argument('-exp_col', type = str, default = 'rationales')
    parser.add_argument('-device', type = str, default = 'cuda')
    parser.add_argument('-input_new_data_path', type = str, default = "../data/unlabeled_data/new_data.csv")
    parser.add_argument('-output_new_data_path', type = str, default = "../data/output_data/new_data.csv")
    parser.add_argument('-saved_model_path', type = str, default = "../data/saved_models/")
    
    parser.add_argument('-full_train', type = str, default = True, 
            help = 'True: full data, False: only use the correct prediction from 1st phase for training in 2nd phase')
    
    parser.add_argument('-mode', type = str, default = 'eval', help='train: train model, eval:evaluate with n-folds, \
                                prediction: make prediction on new data')

   
    label_idx_map = {'disease_signs_or_symptoms':0,  'disease_transmission':1, 'prevention':2, 'treatment':3,
                'deaths_reports':4, 'not_related_or_irrelevant':5}
    prepro_exp = 'prepro_exp'
    prepro_text = 'prepro_text'
    prepro_label = 'prepro_label'
    

    args = parser.parse_args()
    idx_label_map = {idx: label for label, idx in label_idx_map.items()}
    
    if args.mode == 'prediction':
        mtlTrainer = MTLTrainer(args)
        clsTrainer = CLSTrainer(args)
        num_classes = len(label_idx_map)
        print("Load models...")
        mtlTrainer.load(num_classes = num_classes, saved_model_path = args.saved_model_path+args.event_type+"/phase1.pt")
        clsTrainer.load(num_classes = num_classes, saved_model_path = args.saved_model_path+args.event_type+"/phase2.pt")

        print("Read new data...")
        new_data = []
        prepro_data = []
        with open(args.input_new_data_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                new_data.append(line.strip())
                prepro_data.append(utils.preprocess_text(line.strip()))

        cls_pred_p1, _, exp_pred_data, exp_pred_masked, exp_pred_probs = mtlTrainer.classify(prepro_data)
        cls_pred_p2, cls_pred_probs = clsTrainer.classify(exp_pred_masked)

        print("Total time: ", time.time()-begin_time)
       

    else:
        # load data 
        data = pd.read_csv(args.input_path, delimiter = '\t')

        # preprocess exp
        data[prepro_exp] = data[args.exp_col].apply(lambda x: x.strip().replace('[SEP]', args.sep_exp_token))
        data[prepro_exp] = data[prepro_exp].apply(lambda x: utils.preprocess_text(x, lower = True))
        data[prepro_exp] = data[prepro_exp].apply(lambda x: [y.strip() for y in x.split(args.sep_exp_token)])
        # preprocess text
        data[prepro_text] = data[args.text_col].apply(lambda x: utils.preprocess_text(x, lower = True))
        
        data[prepro_label] = data[args.label_col].apply(lambda x: label_idx_map[x])

        data.drop_duplicates(subset = prepro_text, inplace = True)
        print("Data: ", data.shape)

        # initialize models
        text_data = np.array(data[prepro_text])
        exp_data = np.array(data[prepro_exp])
        cls_data = np.array(data[prepro_label])
        

        if args.mode == 'train':
            #train models on entire data
            mtlTrainer = MTLTrainer(args, text_data, cls_data, exp_data)
            print(">>>>> Phase 1...........")
            train_exp_data, train_labels = mtlTrainer.train(saved_model_path = args.saved_model_path+args.event_type+"/phase1.pt")
            print(">>>>> Phase 2...........")
            clsTrainer = CLSTrainer(args, train_exp_data, cls_data)
            clsTrainer.train(saved_model_path = args.saved_model_path + args.event_type+"/phase2.pt")

        elif args.mode == "eval":
            mtlTrainer = MTLTrainer(args, text_data, cls_data, exp_data)
            clsTrainer = CLSTrainer(args, text_data, cls_data)
            #cross-validate 
            kfold = StratifiedShuffleSplit(n_splits = args.n_folds, test_size = args.test_size, random_state = args.random_state)
            fold = 0
            for train_indices, remain_indices in kfold.split(text_data, cls_data):
                valid_indices, test_indices = train_test_split(remain_indices, test_size = 0.5, random_state = args.random_state,
                                    stratify = cls_data[remain_indices])
                print("---------------------FOLD {}-----------------------".format(fold))
                print(">>>>> Phase 1...........")
                train_exp_pred_data, train_labels, valid_exp_pred_data, \
                    valid_labels, test_exp_pred_data, test_labels = mtlTrainer.eval(train_indices, valid_indices, test_indices)
                
                print(">>>>> Phase 2............")
                clsTrainer.eval(train_exp_pred_data, train_labels, valid_exp_pred_data, valid_labels, test_exp_pred_data, test_labels)
                print("--------------------------------------------------")
                fold+=1

    

    



