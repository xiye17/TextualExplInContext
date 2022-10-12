import sys
sys.path.append('.')

import argparse
import os
from os.path import join

from tqdm import tqdm

from utils import *
from dataset_utils import read_snli_data, f1auc_score, train_max_accuracy
from fewshot import result_cache_name as dev_result_cache_name
from fewshot import (    
    post_process_fewshot_prediction
)
from leave_one_fewshot import result_cache_name as train_result_cache_name
from leave_one_fewshot import calib_result_cache_name as calib_result_cache_name

from calib_exps.calib_utils import (
    inspect_confusion_matrix,
    filter_by_predicted_label,
    FewshotClsReranker,
    set_seed
)
from sklearn.metrics import roc_auc_score
import numpy as n

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard") 
    parser.add_argument('--num_shot', type=int, default=32)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_calib', type=int, default=96)
    parser.add_argument('--sub_calib', type=int, default=96)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--run_ub', default=False, action='store_true')
    parser.add_argument('--run_only_calib', default=False, action='store_true')

    args = parser.parse_args()
    specify_engine(args)
    return args

def train_calibrator_from_train_set(examples, predictions):
    reranker = FewshotClsReranker()
    reranker.train(examples, predictions)
    return reranker

def read_examples_and_predictions(args, split):
    if split == 'train':
        train_set = read_snli_data(f"data/train.json")
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
        examples = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]
        predictions = read_json(dev_result_cache_name(args))
    return examples, predictions

def train_fewshot_calibrator(args, split='train'):
    examples, predictions = read_examples_and_predictions(args, split)
    for p in predictions:
        post_process_fewshot_prediction(p, args.style)
    # inspect_confusion_matrix(examples, predictions)
    return train_calibrator_from_train_set(examples, predictions)

def test_fewshot_calibrator(args, reranker, split='dev'):
    examples, predictions = read_examples_and_predictions(args, split)
    for p in predictions:
        post_process_fewshot_prediction(p, args.style)
    # inspect_confusion_matrix(examples, predictions)
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
        reranker = train_fewshot_calibrator(args, 'dev')
    else:
        reranker = train_fewshot_calibrator(args)
    test_fewshot_calibrator(args, reranker)
