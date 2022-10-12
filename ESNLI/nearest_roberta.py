
import argparse
from ast import arg
import os
from tqdm import tqdm

from utils import *
from dataset_utils import read_snli_data, f1auc_score
from comp_utils import safe_completion, length_of_prompt, gpt_style_tokenize
from collections import Counter

import numpy as np
from fewshot import in_context_prediction, calc_fewshot_pred_with_prob, evaluate_nli_predictions
from sklearn.metrics import pairwise_distances

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
    parser.add_argument('--num_train', type=int, default=128)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)
    args = parser.parse_args()
    specify_engine(args)
    return args


def result_cache_name(args):
    return "misc/fewroberta_{}_tr{}-{}_ns{}_dv{}-{}_{}_predictions.json".format(args.engine_name,
        args.train_slice, args.train_slice + args.num_train, args.num_shot, args.dev_slice, args.dev_slice + args.num_dev,
        args.style)


def select_shots_from_train(ex, training_data, num_shot):
    x_emb = np.asarray([ex['embedding']])
    y_embs = np.asarray([e['embedding'] for e in training_data])
    
    distancs = pairwise_distances(x_emb, y_embs).flatten().tolist()    
    # print(len(distancs))
    ranked_examples = sorted(zip(training_data, distancs), key=lambda x: x[1])

    # print("Query", ex['premise'], ex['hypothesis'])
    # for x in ranked_examples[:5]:
    #     x = x[0]
    #     print("\t", x['premise'], x['hypothesis'])

    return [x[0] for x in ranked_examples[:num_shot]]

def test_nn_few_shot_performance(args):
    print("Running prediction")

    train_set = read_snli_data(f"data/train.json")
    train_set = train_set[args.train_slice:(args.train_slice + args.num_train)]

    dev_set = read_snli_data(f"data/dev.json")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    train_embeddings = np.load('cls_vecs/esnli_train_roberta.npy')
    train_embeddings = train_embeddings[args.train_slice:(args.train_slice + args.num_train)]
    for ex, emb in zip (train_set, train_embeddings):
        ex['embedding'] = emb

    dev_embeddings = np.load('cls_vecs/esnli_dev_roberta.npy')
    dev_embeddings = dev_embeddings[args.dev_slice:(args.dev_slice + args.num_dev)]    
    for ex, emb in zip (dev_set, dev_embeddings):
        ex['embedding'] = emb

    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        shots = select_shots_from_train(x, train_set, args.num_shot)
        predictions.append(in_context_prediction(x, shots, engine=args.engine, style=args.style, length_test_only=args.run_length_test))
    
    if args.run_length_test:
        print(result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_COMP_TOKENS)
        return

    # save
    dump_json(predictions, result_cache_name(args))
    # acc
    for p in predictions:
        p['prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    evaluate_nli_predictions(dev_set, predictions)


def analyze_nn_few_shot_performance(args):
    dev_set = read_snli_data(f"data/dev.json")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]    

    predictions = read_json(result_cache_name(args))
    for p in predictions:
        p['prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    evaluate_nli_predictions(dev_set, predictions, do_print=True)
    print(result_cache_name(args))

if __name__ == '__main__':
    args = _parse_args()
    if args.run_prediction:
        test_nn_few_shot_performance(args)
    else:
        analyze_nn_few_shot_performance(args)