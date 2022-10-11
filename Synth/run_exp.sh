#!/bin/bash

# slice choice: 0, 16, 32, 48, 64
TR_SLICE=0

# few-shot exp
python few_shot.py --run_pred --train_slice ${TR_SLICE}

# e-p and p-e
python joint.py --run_pred --style e-p --train_slice ${TR_SLICE}
python joint.py --run_pred --style p-e --train_slice ${TR_SLICE}

#  calibrating p-e
python posthoc_pipeline.py --style p-e --train_slice ${TR_SLICE}
