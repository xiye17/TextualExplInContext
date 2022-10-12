import numpy as np

def get_joint_prompt_helper(style):
    if style == "p-e":
        return JointPandEPromptHelper(style)
    elif style == "e-p":
        return JointEandPPromptHelper(style)
    else:
        raise RuntimeError("Unsupported prompt style")

def normalize_prediction(x):
    x = x.lstrip()
    if x.lower() == 'true': x = 'True'
    if x.lower() == 'false': x = 'False'
    if x.lower() == 'neither': x = 'Neither'
    return x

class JointPrompHelper:
    style = None
    def __init__(self, style):
        self.label_leading_token = None
        self.style = style

    def prompt_for_joint_prediction(self, ex, shots):
        raise NotImplementedError()

    def post_process(self, p):
        self.post_process_prediction(p)
        self.post_process_confidence(p)

    def post_process_confidence(self, pred):
        completion_offset = pred["completion_offset"]
        tokens = pred["logprobs"]["tokens"]
        token_offset = pred["logprobs"]["text_offset"]

        completion_start_tok_idx = token_offset.index(completion_offset)
        # exclusive idxs
        if "<|endoftext|>" in tokens:
            completion_end_tok_idx = tokens.index("<|endoftext|>") + 1
        else:
            completion_end_tok_idx = len(tokens)
        completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
        completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]

        ans_logprob, rat_logprob = self.extract_answer_and_rationale_logprobs(pred, token_offset, completion_start_tok_idx, completion_tokens, completion_probs)

        top_choices = pred["logprobs"]["top_logprobs"][completion_start_tok_idx]
        # print(top_choices)
        cls_probs = []
        for t in self.label_leading_token:
            if t in top_choices:
                cls_probs.append(np.exp(top_choices[t]))
            else:
                cls_probs.append(.0) 
        pred['class_probs'] = cls_probs

        pred["answer_logprob"] = ans_logprob
        pred["rationale_logprob"] = rat_logprob
        pred["joint_logprob"] = ans_logprob + rat_logprob
        return ans_logprob, rat_logprob

    def extract_answer_and_rationale(self, p):
        raise NotImplementedError()

    def post_process_prediction(self, p):
        ans, rat = self.extract_answer_and_rationale(p)
        p["answer"] = ans
        p["rationale"] = rat
        p["label"] = normalize_prediction(ans)
        return ans, rat

    def extract_answer_and_rationale_logprobs(self):
        raise NotImplementedError()

class JointPandEPromptHelper(JointPrompHelper):
    def __init__(self, style):
        super().__init__(style)
        self.sep = ", because"
        self.label_leading_token = [' True', ' False', ' Neither']

    def get_textual_explanation(self, x):
        expl = x["explanations"][0]["text_explanation"]
        expl = expl[0].lower() + expl[1:]
        return expl

    def prompt_for_joint_prediction(self, ex, shots):
        stop_signal = "\n\n"
        showcase_examples = [
            "{}\nQ: {} True, False, or Neither?\nA: {}{} {}\n".format(s["premise"], s["hypothesis"], 
            s["label"], self.sep, self.get_textual_explanation(s)) for s in shots
        ]
        input_example = "{}\nQ: {} True, False, or Neither?\nA:".format(ex["premise"], ex["hypothesis"])

        prompt = "\n".join(showcase_examples + [input_example])
        return prompt, stop_signal

    def extract_answer_and_rationale_logprobs(self, pred, token_offset, completion_start_tok_idx, completion_tokens, completion_probs):
        if self.sep in pred["text"]:
            sep_token_offset = pred["completion_offset"] + pred["text"].index(self.sep)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            ans_logprob = sum(completion_probs[:sep_start_idx + 2])
            rat_logprob = sum(completion_probs[(sep_start_idx + 2):])            
        else:            
            ans_logprob = float("-inf")
            rat_logprob = float("-inf")
        return ans_logprob, rat_logprob

    def extract_answer_and_rationale(self, p):
        text = p["text"].strip()        
        segments = text.split(self.sep)   
        answer = segments[0].strip()
        rationale = segments[1].strip()

        return answer, rationale


class JointEandPPromptHelper(JointPrompHelper):
    def __init__(self, style):
        super().__init__(style)
        self.sep = 'The answer is'
        self.label_leading_token = [' True', ' False', ' Neither']

    def get_textual_explanation(self, x):
        expl = x["explanations"][0]["text_explanation"]
        expl = expl[0].lower() + expl[1:]
        if expl[-1] == ',':
            expl = expl[:-1] + '.'
        elif expl[-1] != '.':
            expl = expl + '.'
        return expl

    def prompt_for_joint_prediction(self, ex, shots):
        stop_signal = "\n\n"
        showcase_examples = [
            "{}\nQ: {} True, False, or Neither?\nA: Because {} Answer is {}.\n".format(s["premise"], s["hypothesis"], 
           self.get_textual_explanation(s), s["label"]) for s in shots
        ]
        input_example = "{}\nQ: {} True, False, or Neither?\nA: Because".format(ex["premise"], ex["hypothesis"])

        prompt = "\n".join(showcase_examples + [input_example])
        return prompt, stop_signal

    def extract_answer_and_rationale_logprobs(self, pred, token_offset, completion_start_tok_idx, completion_tokens, completion_probs):
        text = pred["text"]                
        if ' Answer is' in text:
            sep = ' Answer is'
        elif ' answer is' in text:
            sep = ' answer is'
        else:            
            return -100, -100
        sep_token_offset = pred["completion_offset"] + pred["text"].index(sep)
        sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx


        rat_logprob = sum(completion_probs[:sep_start_idx + 3])
        ans_logprob = sum(completion_probs[(sep_start_idx + 3):])
        return ans_logprob, rat_logprob

    def post_process_confidence(self, pred):
        if 'logprobs' not in pred:
            pred["answer_logprob"] = -100
            pred["rationale_logprob"] = -100
            pred["joint_logprob"] = -100
            return -100, -100
        return super().post_process_confidence(pred)

    def extract_answer_and_rationale(self, p):
        text = p["text"].strip()
        # print(text)
        if ' Answer is' in text:
            sep = ' Answer is'
        elif ' answer is' in text:
            sep = ' answer is'
        else:
            print('unparse', text)
            answer = 'Neither'
            rationale = 'none'
            return answer, rationale
        segments = text.split(sep)
        rationale = segments[0].strip()
        answer = segments[1].strip().rstrip('.')
        return answer, rationale

