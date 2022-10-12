import pickle
import json

from matplotlib import lines

def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def read_jsonlines(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [json.loads(x) for x in lines]

def add_engine_argumenet(parser):
    parser.add_argument('--engine',
                            default='text-davinci-001',
                            choices=['davinci', 'text-davinci-001', 'text-davinci-002'])

# historical code
def specify_engine(args):
    args.engine_name = args.engine


