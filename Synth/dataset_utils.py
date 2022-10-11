from utils import *

def read_synth_data(filename):
    # return (text_rationale,question,context,answer)
    data = read_json(filename)["data"]

    examples = []
    for d in data:
        t_rat = d["title"]
        par = d["paragraphs"][0]
        c = par["context"]
        qa = par["qas"][0]
        q = qa["question"]
        a = qa["answers"][0]["text"]

        # remake rationale
        decimal = t_rat.index('.')        
        t_rat = t_rat[:decimal] + ' and' + t_rat[decimal + 1:]        
        ex = {
            "text_rationale": t_rat,
            "question": q,
            "context": c,
            "answer": a,
        }

        examples.append(ex)

    return examples

def index_example(ex):
    context = ex["context"]
    context = context.replace(" a ", " ").replace(" an ", " ")
    clues = context.split(".")
    tokens = set()
    for c in clues:
        if not c:
            continue
        segs = c.split(maxsplit=1)        
        tokens.add(segs[0])
        segs = segs[1].rsplit(maxsplit=1)        
        tokens.add(segs[0])
        tokens.add(segs[1])
            
    # tokens = set(context.replace(".", "").split())
    ex["words"] = tokens
    return ex

def reorder_rationale(ex):
    context = ex["context"]
    text_rationale = ex["text_rationale"]

    sent_boundary = text_rationale.index('.')
    fst_sent = text_rationale[:(sent_boundary + 1)]
    snd_sent = text_rationale[(sent_boundary + 2):]

    fst_pos = context.index(fst_sent)
    snd_pos = context.index(snd_sent)

    # in original order
    if fst_pos < snd_pos:
        return ex
    text_rationale = snd_sent + ' ' + fst_sent    

    ex["text_rationale"] = text_rationale
    return ex
