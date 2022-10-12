import argparse
import os
from tqdm import tqdm

from utils import *
from dataset_utils import read_snli_data, f1auc_score
from comp_utils import safe_completion, length_of_prompt
from collections import Counter

from fewshot import (
    in_context_prediction,
    evaluate_nli_predictions,
    calc_fewshot_pred_with_prob
)

import numpy as np

_MAX_COMP_TOKENS = 12

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')    
    parser.add_argument('--num_shot', type=int, default=32)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_calib', type=int, default=96)
    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/crossfewshot_{}_tr{}-{}_{}_predictions.json".format( args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.style)

def calib_result_cache_name(args):
    return "misc/crossfewshotcalib_{}_tr{}-{}_cal{}_{}_predictions.json".format( args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.num_calib, args.style)

def test_leave_one_fewshot(args):
    print("Running prediction")

    train_set = read_snli_data(f"data/train.json")
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]

    predictions = []
    for i, x in tqdm(enumerate(train_set), total=len(train_set), desc="Predicting"):
        i_train_set = []
        predictions.append(in_context_prediction(x, 
            [y for (j, y) in enumerate(train_set) if j != i],
            args.engine, style=args.style, length_test_only=args.run_length_test))

    if args.run_length_test:
        print(result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_COMP_TOKENS)
        return

    # save
    dump_json(predictions, result_cache_name(args))
    # acc
    for p in predictions:
        p['prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    evaluate_nli_predictions(train_set, predictions)

def test_calib_fewshot(args):
    print("Running prediction")

    train_set = read_snli_data(f"data/train.json")
    dev_set = train_set[(args.train_slice + args.num_shot):(args.train_slice + args.num_shot + args.num_calib)]
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    

    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        predictions.append(in_context_prediction(x, train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test))

    if args.run_length_test:
        print(calib_result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_COMP_TOKENS)
        return

    # save
    dump_json(predictions, calib_result_cache_name(args))
    # acc
    for p in predictions:
        p['prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    evaluate_nli_predictions(dev_set, predictions)

def analyze_leave_one_fewshot(args):
    train_set = read_snli_data(f"data/train.json")
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]

    predictions = read_json(result_cache_name(args))
    for p in predictions:
        p['prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    evaluate_nli_predictions(train_set, predictions)
    print(result_cache_name(args))

def analyze_calib_fewshot(args):
    train_set = read_snli_data(f"data/train.json")
    train_set = train_set[(args.train_slice + args.num_shot):(args.train_slice + args.num_shot + args.num_calib)]

    predictions = read_json(calib_result_cache_name(args))
    for p in predictions:
        p['prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    evaluate_nli_predictions(train_set, predictions)
    print(calib_result_cache_name(args))

if __name__ == '__main__':
    args = _parse_args()
    if args.run_prediction:
        test_leave_one_fewshot(args)
        test_calib_fewshot(args)
    else:
        analyze_leave_one_fewshot(args)
        analyze_calib_fewshot(args)
