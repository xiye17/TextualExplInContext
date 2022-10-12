import sys
sys.path.append('.')

import argparse
import os
from os.path import join

from tqdm import tqdm

from utils import *
from dataset_utils import read_snli_data, f1auc_score, train_max_accuracy
from prompt_helper import get_joint_prompt_helper
from collections import Counter

from leave_one_joint import result_cache_name as train_result_cache_name
from leave_one_joint import calib_result_cache_name as calib_result_cache_name
from joint import result_cache_name as dev_result_cache_name

import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

from calib_exps.calib_utils import set_seed

class RobertaReranker:
    def __init__(self):
        self.model = LogisticRegression(C=10, fit_intercept=True)
        self.index_of_label = {'True': 0, 'False': 1, 'Neither': 2}
        self.label_of_index = ['True', 'False', 'Neither']

    def train(self, examples, predictions):
        # calibrate true
        orig_preds = np.asarray([self.index_of_label[p['label']] for p in predictions])
        gt_scores = np.asarray([self.index_of_label[ex['label']] for ex in examples])

        training_feature = np.asarray([x['embedding'] for x in examples])

        self.model.fit(training_feature, gt_scores)
        train_preds = self.model.predict(training_feature)
        train_acc = np.mean(train_preds ==gt_scores)
        # print("Base ACC: {:.2f}".format(np.mean(orig_preds == gt_scores) * 100), "Train ACC: {:.2f}".format(train_acc * 100))

    def apply(self, ex, pred):
        p = self.model.predict(np.array([ex['embedding']]))[0]
        return self.label_of_index[p]


class RobertaJointReranker:
    def __init__(self):
        self.model = LogisticRegression(C=10, fit_intercept=True)
        self.index_of_label = {'True': 0, 'False': 1, 'Neither': 2}
        self.label_of_index = ['True', 'False', 'Neither']
        self.norm = 1.0

    def train(self, examples, predictions):
        # calibrate true
        orig_preds = np.asarray([self.index_of_label[p['label']] for p in predictions])
        gt_scores = np.asarray([self.index_of_label[ex['label']] for ex in examples])
        cls_scores = np.asarray([p['class_probs'] for p in predictions])        
        embedings = np.array([x['embedding'] for x in examples])        
        training_feature = np.concatenate((cls_scores, embedings), axis=1)

        self.model.fit(training_feature, gt_scores) 

        train_preds = self.model.predict(training_feature)
        train_acc = np.mean(train_preds ==gt_scores)
        # print("Base ACC: {:.2f}".format(np.mean(orig_preds == gt_scores) * 100), "Train ACC: {:.2f}".format(train_acc * 100))

    def apply(self, ex, pred):
        emb = ex['embedding']
        probs = np.array(pred['class_probs'])
        feature = np.concatenate((probs, emb)).reshape((1,-1))        
        p = self.model.predict(feature)[0]
        return self.label_of_index[p]


def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="p-e")    
    parser.add_argument('--num_shot', type=int, default=32)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_calib', type=int, default=96, help='used for indexing additional calibration data')
    parser.add_argument('--sub_calib', type=int, default=96, help='the actual calibration instance size')
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)


    parser.add_argument('--run_ub', default=False, action='store_true', help='train calibrator on dev features')
    parser.add_argument('--run_only_calib', default=False, action='store_true', help='do not include leave-one-out samples from the training')
    parser.add_argument('--do_conf', default=False, action='store_true', help='confidence only reranker')

    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def read_examples_and_predictions(args, split):
    if split == 'train':
        train_set = read_snli_data(f"data/train.json")
        train_embeddings = np.load('cls_vecs/esnli_train_roberta.npy')
        for ex, emb in zip (train_set, train_embeddings):
            ex['embedding'] = emb

        examples = train_set[args.train_slice:(args.train_slice + args.num_shot)]
        predictions = read_json(train_result_cache_name(args))
        calib_examples = train_set[(args.train_slice + args.num_shot):(args.train_slice + args.num_shot + args.num_calib)]
        calib_predictions = read_json(calib_result_cache_name(args))
        if args.run_only_calib:
            assert args.sub_calib > 0
            examples = calib_examples[:args.sub_calib]
            predictions = calib_predictions[:args.sub_calib]
        elif args.sub_calib > 0:
            calib_examples = calib_examples[:args.sub_calib]
            calib_predictions = calib_predictions[:args.sub_calib]
            examples = examples + calib_examples
            predictions = predictions + calib_predictions
    else:
        dev_set = read_snli_data(f"data/dev.json")
        dev_embeddings = np.load('cls_vecs/esnli_dev_roberta.npy')
        for ex, emb in zip (dev_set, dev_embeddings):
            ex['embedding'] = emb

        examples = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]
        predictions = read_json(dev_result_cache_name(args))

    return examples, predictions

def train_roberta_calibrator(args, split='train'):
    examples, predictions = read_examples_and_predictions(args, split)
    [args.helper.post_process(p) for p in predictions]        

    reranker = RobertaJointReranker()
    reranker.train(examples, predictions)
    return reranker

def test_roberta_calibrator(args, reranker, split='dev'):
    examples, predictions = read_examples_and_predictions(args, split)
    [args.helper.post_process(p) for p in predictions]        

    gts = [x['label'] for x in examples]
    original_predictions = [x['label'] for x in predictions]
    calibrated_predictions = [reranker.apply(ex, pred) for (ex, pred) in zip(examples, predictions)]

    calc_acc_score = lambda a,b: sum([x == y for (x,y) in zip(a,b)]) * 1.0 / len(a)
    print("Orig ACC {:.2f}".format(calc_acc_score(gts, original_predictions) * 100))
    print("Calib ACC {:.2f}".format(calc_acc_score(gts, calibrated_predictions) * 100))


if __name__=='__main__':
    args = _parse_args()
    set_seed(42)
    if args.run_ub:
        reranker = train_roberta_calibrator(args, 'dev')
    else:
        reranker = train_roberta_calibrator(args)
    test_roberta_calibrator(args, reranker)