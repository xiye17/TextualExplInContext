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

from calib_exps.calib_utils import (
    inspect_confusion_matrix,
    FewshotClsReranker,
    JointClsProbExplReranker,
    set_seed
)

nlp = spacy.load('en_core_web_sm')

def mkdir_p(d):
    if not os.path.exists(d):
        os.mkdir(d)

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

def extract_stem_tokens(text):
    doc = nlp(text)
    stem_tokens = []
    for i, t in enumerate(doc):
        pos, tag = t.pos_, t.tag_
        if pos == 'AUX':
            continue
        is_stem = False
        if tag.startswith('NN'):
            is_stem = True
        if tag.startswith('VB'):
            is_stem = True
        if tag.startswith('JJ'):
            is_stem = True
        if tag.startswith('RB'):
            is_stem = True
        if tag == 'CD':
            is_stem = True
        if is_stem:
            stem_tokens.append({
                'index': i,
                'text': t.text,
                'lemma': t.lemma_,
                'pos': t.pos_,
                'tag': t.tag_
            })
    return stem_tokens


def rationale_coverage_quality(r, q):
    q_stem_tokens = extract_stem_tokens(q)
    r_stem_tokens = extract_stem_tokens(r)
    r_lemma_tokens = [x['lemma'] for x in r_stem_tokens]
    q_lemma_tokens = [x['lemma'] for x in q_stem_tokens]

    hit = 0
    for t in r_lemma_tokens:
        if t in q_lemma_tokens:
            hit += 1

    return hit / len(r_lemma_tokens)

def process_rationale_quality_feature(ex, pred):
    premise_coverage = rationale_coverage_quality(pred['rationale'], ex['premise'])
    hypothesis_coverage = rationale_coverage_quality(pred['rationale'], ex['hypothesis'])
    pred['premise_coverage'] = premise_coverage
    pred['hypothesis_coverage'] = hypothesis_coverage


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

def train_joint_calibrator(args, split='train'):
    examples, predictions = read_examples_and_predictions(args, split)
    [args.helper.post_process(p) for p in predictions]
    if args.do_conf:
        reranker = FewshotClsReranker()
    else:
        [process_rationale_quality_feature(ex, pred) for (ex, pred) in zip(examples, predictions)]
        reranker = JointClsProbExplReranker()
    # inspect_confusion_matrix(examples, predictions)
    reranker.train(examples, predictions)
    return reranker

def test_joint_calibrator(args, reranker, split='dev'):
    examples, predictions = read_examples_and_predictions(args, split)

    [args.helper.post_process(p) for p in predictions]
    if not args.do_conf:
        [process_rationale_quality_feature(ex, pred) for (ex, pred) in zip(examples, predictions)]
    # inspect_confusion_matrix(examples, predictions)
        
    gts = [x['label'] for x in examples]
    original_predictions = [x['label'] for x in predictions]
    calibrated_predictions = [reranker.apply(ex, pred) for (ex, pred) in zip(examples, predictions)]

    calc_acc_score = lambda a,b: sum([x == y for (x,y) in zip(a,b)]) * 1.0 / len(a)
    print("Orig ACC {:.2f}".format(calc_acc_score(gts, original_predictions) * 100))
    print("Calib ACC {:.2f}".format(calc_acc_score(gts, calibrated_predictions) * 100))

def preprare_for_annotation(args, split='dev'):
    examples, predictions = read_examples_and_predictions(args, split)

    [args.helper.post_process(p) for p in predictions]
    if not args.do_conf:
        [process_rationale_quality_feature(ex, pred) for (ex, pred) in zip(examples, predictions)]

    for idx, (ex, pred) in enumerate(zip(examples, predictions)):
        gt = ex["label"]
        p = pred["label"]
        print("--------------{} EX {} --------------".format(idx, gt == p))
        print('PRE: {:.2f}'.format(pred['premise_coverage']), 'PROB: {:.2f}'.format(pred['hypothesis_coverage']))
        print(pred["prompt"].split('\n\n')[-1])
        print('P:', p, 'G:', gt)
        print('RAT:', pred['rationale'])
        print('FACT:', 'N')
        print('CONT:', 'N')
        print('RELA:', 'N')

if __name__=='__main__':
    args = _parse_args()

    if args.run_ub:
        reranker = train_joint_calibrator(args, 'dev')
    else:
        reranker = train_joint_calibrator(args)
    test_joint_calibrator(args, reranker)
