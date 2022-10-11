import argparse
import os
import subprocess

def run_cmd(cmds, is_pred_task=False):
    if is_pred_task:
        cmds.append('--run_pred')
    print("Running CMD")
    print(" ".join(cmds))
    output = subprocess.check_output(cmds, stderr=subprocess.DEVNULL)
    output = output.decode()
    return output

def fewshot_exp(train_slice):
    print('Few Shot', 'slice', train_slice)
    cmds = ['python', 'few_shot.py', '--train_slice', train_slice, '--style', 'standard', '--show_result']
    output = run_cmd(cmds, True)
    print(output)

def fewshotnn_exp(train_slice):
    print('Few Shot NN', 'slice', train_slice)
    cmds = ['python', 'fewshot_roberta.py', '--train_slice', train_slice, '--style', 'standard', '--show_result']
    output = run_cmd(cmds, True)
    print(output)

def joint_exp(train_slice):
    print('E-P', 'slice', train_slice)
    cmds = ['python', 'manual_joint.py', '--train_slice', train_slice, '--style', 'e-p', '--show_result']
    output = run_cmd(cmds, True)
    print(output)

    print('P-E', 'slice', train_slice)
    cmds = ['python', 'manual_joint.py', '--train_slice', train_slice, '--style', 'p-e', '--show_result']
    output = run_cmd(cmds, True)
    print(output)

def calib_exp(train_slice):
    print('E-P + Expl Calib', 'slice', train_slice)
    cmds = ['python', 'calib_exps/run_exp.py', '--train_slice', train_slice, '--style', 'e-p']
    output = run_cmd(cmds)

    print(output)

def calib_roberta(train_slice):
    print('E-P + Zhang', 'slice', train_slice)
    cmds = ['python', 'calib_exps/run_roberta.py', '--train_slice', train_slice, '--style', 'e-p']
    output = run_cmd(cmds)
    print(output)

if __name__=='__main__':
    for c in [0, 6,12,18,24]:
        c = str(c)
        fewshot_exp(c)
        fewshotnn_exp(c)
        joint_exp(c)
        calib_exp(c)
        calib_roberta(c)
