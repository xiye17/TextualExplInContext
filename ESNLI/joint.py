import argparse
import os
from tqdm import tqdm

from utils import *
from dataset_utils import read_snli_data, f1auc_score
from comp_utils import safe_completion, length_of_prompt
from prompt_helper import get_joint_prompt_helper
from fewshot import normalize_prediction
from collections import Counter

import numpy as np

_MAX_COMP_TOKENS = 70

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="p-e")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')    
    parser.add_argument('--num_shot', type=int, default=32)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)
    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/joint_{}_tr{}-{}_dv{}-{}_{}_predictions.json".format(args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.dev_slice + args.num_dev,
        args.style)

def in_context_joint_prediction(ex, training_data, engine, prompt_helper, length_test_only=False):
    prompt, stop_signal = prompt_helper.prompt_for_joint_prediction(ex, training_data)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_COMP_TOKENS)
        print("-----------------------------------------")
        print(pred)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, _MAX_COMP_TOKENS, stop_signal, temp=0.0, logprobs=5)        

    pred["id"] = ex["id"]
    pred["prompt"] = prompt
    if len(pred["text"]) > len(prompt):
        pred["text"] = pred["text"][len(prompt):]
    else:
        pred["text"] = "null"
    pred["completion_offset"] = len(prompt)
    return pred

def evaluate_joint_nli_predictions(dev_set, predictions, do_print=False):        
    acc_records = []
    all_probs = []
    all_texts = []
    for ex, pred in zip(dev_set, predictions):
        gt = ex["label"]
        orig_p = pred["answer"]
        p = normalize_prediction(orig_p)
        all_texts.append(p)
        ex = p == gt        
        acc_records.append(ex)        
        all_probs.append(pred['answer_logprob'])

        if do_print:
            print("--------------EX {}--------------".format(ex))
            print(pred["prompt"].split('\n\n')[-1])
            # print('RAW:' + pred["text"])
            print('P:', p, 'G:', gt)
            print('P RAT:', pred['rationale'])
            # print('Reference RAT:', ex["explanations"][0]['rationale'])
            # print('ID:', pred['id'])

    print("ACC", sum(acc_records) / len(acc_records))

def test_joint_performance(args):
    print("Running prediction")

    train_set = read_snli_data(f"data/train.json")
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]

    dev_set = read_snli_data(f"data/dev.json")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        predictions = []
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            predictions.append(in_context_joint_prediction(x, train_set, args.engine, args.helper, length_test_only=args.run_length_test))

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'OVER', sum([x > 2048 for x in predictions]))
            return
        # save
        dump_json(predictions, result_cache_name(args))
    [args.helper.post_process(p) for p in predictions]

    # acc
    evaluate_joint_nli_predictions(dev_set, predictions)
    print(result_cache_name(args))

def analyze_joint_performance(args):
    dev_set = read_snli_data(f"data/dev.json")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    predictions = read_json(result_cache_name(args))
    [args.helper.post_process(p) for p in predictions]
    evaluate_joint_nli_predictions(dev_set, predictions, do_print=False)
    print(result_cache_name(args))

if __name__ == '__main__':
    args = _parse_args()
    if args.run_prediction:
        test_joint_performance(args)
    else:
        analyze_joint_performance(args)
