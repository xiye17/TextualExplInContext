
import argparse
import os
from tqdm import tqdm

from utils import *
from dataset_utils import read_snli_data, f1auc_score
from comp_utils import safe_completion, length_of_prompt
from collections import Counter

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
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)
    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/few_{}_tr{}-{}_dv{}-{}_{}_predictions.json".format(args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.dev_slice + args.num_dev,
        args.style)

def in_context_prediction(ex, shots, engine, style="standard", length_test_only=False):
    if style == "standard":
        showcase_examples = [
            "{}\nQ: {} True, False, or Neither?\nA: {}\n".format(s["premise"], s["hypothesis"], s["label"]) for s in shots
        ]
        input_example = "{}\nQ: {} True, False, or Neither?\nA:".format(ex["premise"], ex["hypothesis"])
        
        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style " + style)

    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_COMP_TOKENS)
        print("-----------------------------------------")
        print(pred)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, _MAX_COMP_TOKENS, '\n', temp=0.0, logprobs=5)    

    pred["id"] = ex["id"]
    pred["prompt"] = prompt
    if len(pred["text"]) > len(prompt):
        pred["text"] = pred["text"][len(prompt):]
    else:
        pred["text"] = "null"
    return pred

def normalize_prediction(x):
    x = x.lstrip()
    if x.lower() == 'true': x = 'True'
    if x.lower() == 'false': x = 'False' 
    if x.lower() == 'neither': x = 'Neither'
    return x

def evaluate_nli_predictions(dev_set, predictions, do_print=False):        
    acc_records = []
    all_probs = []
    all_texts = []
    for ex, pred in zip(dev_set, predictions):
        gt = ex["label"]
        orig_p = pred["text"]
        p = normalize_prediction(orig_p)
        all_texts.append(p)
        ex = p == gt        
        acc_records.append(ex)        
        all_probs.append(pred['prob'])

        if do_print:
            print("--------------EX {}--------------".format(ex))
            print(pred["prompt"].split('\n\n')[-1])
            print('RAW:', orig_p)
            print('P:', p, 'G:', gt)
            print('ID:', pred['id'])
    
    print("ACC", sum(acc_records) / len(acc_records))    

def calc_fewshot_pred_with_prob(pred, style):
    if pred['text'] == "null" or pred['text'] == "overflow":
        return .0
    completion_offset = len(pred["prompt"])
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    completion_end_tok_idx = tokens.index("<|endoftext|>") + 1 if '<|endoftext|>' in tokens else len(tokens)
    # completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]    
    ans_logprob = sum(completion_probs)
    
    return np.exp(ans_logprob)

def calc_fewshot_cls_prob(pred, style):
    if pred['text'] == "null" or pred['text'] == "overflow":
        pred['class_probs'] = [.0, .0, 1.0]
    completion_offset = len(pred["prompt"])
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    completion_end_tok_idx = tokens.index("<|endoftext|>") + 1 if '<|endoftext|>' in tokens else len(tokens)
    # completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]

    top_choices = pred["logprobs"]["top_logprobs"][completion_start_tok_idx]    
    if style == 'standard':
        mappings = [' True', ' False', ' Neither']
    else:
        raise RuntimeError("Unsupported Style")
    cls_probs = []
    for t in mappings:
        if t in top_choices:
            cls_probs.append(np.exp(top_choices[t]))
        else:
            cls_probs.append(.0)    
    pred['class_probs'] = cls_probs

def post_process_fewshot_prediction(p, style):
    p['prob'] = calc_fewshot_pred_with_prob(p, style)
    calc_fewshot_cls_prob(p, style)
    p['label'] = normalize_prediction(p['text'])
    

def test_few_shot_performance(args):
    print("Running prediction")

    train_set = read_snli_data(f"data/train.json")
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]

    dev_set = read_snli_data(f"data/dev.json")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        predictions.append(in_context_prediction(x, train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test))

    if args.run_length_test:
        print(result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_COMP_TOKENS)
        return

    # save
    dump_json(predictions, result_cache_name(args))

    print(result_cache_name(args))
    # acc
    for p in predictions:
        post_process_fewshot_prediction(p, args.style)
    evaluate_nli_predictions(dev_set, predictions)


def analyze_few_shot_performance(args):
    dev_set = read_snli_data(f"data/dev.json")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]    

    predictions = read_json(result_cache_name(args))
    for p in predictions:
        post_process_fewshot_prediction(p, args.style)
    evaluate_nli_predictions(dev_set, predictions, do_print=False)
    print(result_cache_name(args))

if __name__ == '__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        analyze_few_shot_performance(args)