# TextualExplInContext
The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning (NeurIPS 2022)

## Setup

Requirements:

* openai
* transformers
* scikit-learn

Set OPENAI KEY:

`export OPENAI_API_KEY=yourkey`

## Run Experiments on Synth
Please run the following commands under `synth` dir.

(1) Running standard few-shot prompting:

`python few_shot.py --run_pred --train_slice 0`


(2) Runninng explain-predict or predict-explain pipeline:

`python joint.py --run_pred --style e-p (p-e) --train_slice 0`

(3) Calibrating the predictions of predict-explain:
`python posthoc_pipeline.py --style p-e --train_slice 0`


**Arguments**

Note that we will store the API query results as json files in `synth/misc`.

There are some arguements in `few_shot.py` and `joint.py`.
* `--engine` specifies the OPENAI API engine (e.g., davinci,text-davinci-001)
* `--style` specifies the prompt format style
* `--train_slice` specifies the random split of the shots in prompts. Because the trainning data is randomly sampled and permuted, different `train_slice` will result in different randomly sampled shot sets.


## Run Experiments on HotpotAdv and E-SNLI
Please refer to `HotpotAdv/run_exp.py` and `ESNL/run_exp.py`. The commands and arguments are the same as the experiments on Synth.