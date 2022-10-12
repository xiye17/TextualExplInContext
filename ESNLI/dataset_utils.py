import sys

from utils import *
import numpy as np


_LABLE_MAPPING = {
    'entailment': 'True',
    'contradiction': 'False',
    'neutral': 'Neither',

}
def read_snli_data(filename):
    data = read_json(filename)
    for d in data:
        d['label'] = _LABLE_MAPPING[d['label']]
    return data

def f1auc_score(score, f1):
    score = np.array(score)
    f1 = np.array(f1)
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    segment = min(1000, score.size - 1)
    T = np.arange(segment) + 1
    T = T/segment
    results = np.array([np.mean(f1[:int(num_test * t)])  for t in T])
    # print(results)
    return np.mean(results)

def train_max_accuracy(x, y):
    x = x.flatten()
    best_acc = 0
    best_v = 0
    for v in x:
        p = x > v
        ac = np.sum(p == y) / y.size
        if ac > best_acc:
            best_acc = ac
            best_v= v
    return best_acc, best_v

def merge_predication_chunks(file1, file2):
    print(file1)
    print(file2)

    file1_args = file1.split("_")
    file2_args = file2.split("_")

    assert len(file1_args) == len(file2_args)

    merged_args = []
    for a1, a2 in zip(file1_args, file2_args):
        if a1.startswith("dv") and a2.startswith("dv"):
            assert a1[2:].split("-")[1] == a2[2:].split("-")[0]
            new_a = a1.split("-")[0] + "-" + a2.split("-")[1]
            merged_args.append(new_a)
        else:
            assert a1 == a2
            merged_args.append(a1)    
    merged_filename = "_".join(merged_args)
    print(merged_filename)
    
    p1 = read_json(file1)
    p2 = read_json(file2)
    dump_json(p1 + p2, merged_filename)    

if __name__ == '__main__':
    merge_predication_chunks(sys.argv[1], sys.argv[2])