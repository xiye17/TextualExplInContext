
import argparse
import os
from tqdm import tqdm

from utils import *
from dataset_utils import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score, read_incorrect_answers
from comp_utils import safe_completion, length_of_prompt
import numpy as np
from sklearn.metrics import pairwise_distances
from few_shot import in_context_prediction, calc_fewshot_pred_with_prob, evaluate_few_shot_predictions, analyze_few_shot_performance


TEST_PART = 250

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308) # firs 58 for calibrating, last 250 for testing
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    args = parser.parse_args()
    specify_engine(args)
    return args


def load_raw_data_ids(fname):
    data = read_json(fname)
    ids = [x["qas_id"] for x in data]
    return ids

def load_dist_dict():
    train_ids = load_raw_data_ids(f"data/sim_train.json")
    dev_ids = load_raw_data_ids(f"data/sim_dev.json")

    train_embeddings = np.load('cls_vecs/hpqa_train_roberta.npy')
    dev_embeddings = np.load('cls_vecs/hpqa_dev_roberta.npy')

    print(len(train_ids), len(train_embeddings))
    print(len(dev_ids), len(dev_embeddings))

    allowed_train_embeddings = np.concatenate((train_embeddings, dev_embeddings[:-TEST_PART]))
    allowed_train_ids = np.concatenate((train_ids, dev_ids[:-TEST_PART]))
    # print(len(allowed_train_ids), allowed_train_embeddings.shape, len(dev_ids[:-TEST_PART]))

    pair_dists = pairwise_distances(dev_embeddings, allowed_train_embeddings)

    dist_dict = {}
    for qid, dists in zip(dev_ids, pair_dists):
        
        dist_dict[qid] = dict(zip(allowed_train_ids, dists))
    return dist_dict

def result_cache_name(args):
    return "misc/fewroberat_an{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.json".format(args.annotation,
        args.engine_name, args.train_slice, args.train_slice + args.num_shot,
        args.dev_slice, args.dev_slice + args.num_dev,args.num_distractor, args.style)

def select_shots(ex, train_set, dev_set, dist_dict, num_shot):
    avaliable_shots = train_set + dev_set[:-TEST_PART]
    all_dists = dist_dict[ex['id']]
    distancs = [all_dists[i['id']] for i in avaliable_shots]    
    ranked_examples = sorted(zip(avaliable_shots, distancs), key=lambda x: x[1])

    return [x[0] for x in ranked_examples[:num_shot]]

def cache_key_func(ex_id, shot_ids):
    return '-'.join([ex_id] + list(shot_ids))

def process_history_predctioins(history, fname):
    predictions = read_json(fname)
    for p in predictions:
        k = cache_key_func(p['id'], p['shot_ids'])
        if k not in history:
            history[k] = p

def get_history_cache():
    history = {}
    print('NUM Record', len(history))
    return history

def test_few_shot_performance(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", n_dist=args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    full_dev_set = dev_set[:]
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    dist_dict = load_dist_dict()
    history = get_history_cache()
    predictions = []

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            train_shots = select_shots(x, train_set, full_dev_set, dist_dict, args.num_shot)
            if args.run_length_test:
                pred = in_context_prediction(x, train_shots, engine=args.engine, style=args.style, length_test_only=args.run_length_test)
            else:
                selected_ids = tuple([y['id'] for y in train_shots])
                query_key = cache_key_func(x['id'], selected_ids)
                if query_key in history:
                    print('Cache Captured', x['question'])
                    pred = history[query_key]
                else:
                    pred = in_context_prediction(x, train_shots, engine=args.engine, style=args.style, length_test_only=args.run_length_test)
                    pred['shot_ids'] = selected_ids
            predictions.append(pred)

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', 32)
            return
        # save
        print('LEN', len(predictions))
        dump_json(predictions, result_cache_name(args))
    # acc
    analyze_few_shot_performance(dev_set, predictions)

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        analyze_few_shot_performance(args)
