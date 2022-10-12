import argparse
import os
import subprocess

FEWSHOT_STYLE = 'standard'
JOINT_STYLE = 'p-e'

def _parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')    
    parser.add_argument('--num_shot', type=int, default=32)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_train', type=int, default=128)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--run_only_calib', default=False, action='store_true')    
    args = parser.parse_args()
    return args

def run_cmd(args, cmds, is_pred_task=False):
    cmds.append('--train_slice')
    cmds.append(str(args.train_slice))

    if is_pred_task:
        if args.run_prediction:
            cmds.append('--run_prediction')
        if args.run_length_test:
            cmds.append('--run_length_test')
    # print("RUN CMD")
    # print(" ".join(cmds))
    # return "\n\n\n\n"
    output = subprocess.check_output(cmds, stderr=subprocess.DEVNULL)
    output = output.decode()
    return output

def main_exp_group(args):
    # few shot
    print("Few Shot")
    output = run_cmd(args, ['python', 'fewshot.py', '--style', FEWSHOT_STYLE], is_pred_task=True)
    print(output.split("\n")[-3])

    # few shot
    print("E-P")
    output = run_cmd(args, ['python', 'joint.py', '--style', 'e-p'], is_pred_task=True)
    print(output.split("\n")[-3])

    # few shot
    print("P-E")
    output = run_cmd(args, ['python', 'joint.py', '--style', 'p-e'], is_pred_task=True)
    print(output.split("\n")[-3])


    # inference on additioinal data for calibrating few-shot performance
    output = run_cmd(args, ['python', 'leave_one_fewshot.py', '--style', FEWSHOT_STYLE], is_pred_task=True)
    # few shot and conf rank
    print("Few Shot Conf Rerank")
    for n in [0, 32, 64, 96]:
        output = run_cmd(args, ['python', 'calib_exps/calib_fewshot.py', '--style', FEWSHOT_STYLE, '--sub_calib', str(n)])
        print(str(n + 32) + ":", output.split("\n")[-2].split(" ")[-1], end=" | ")
    print("")

    # inference on additioinal data for calibrating p-e performance
    output = run_cmd(args, ['python', 'leave_one_joint.py', '--style', JOINT_STYLE], is_pred_task=True)
    # p-e conf rerank
    print("P-E Conf")
    for n in [0, 32, 64, 96]:    
        output = run_cmd(args, ['python', 'calib_exps/calib_joint.py', '--style', JOINT_STYLE, '--do_conf', '--sub_calib', str(n)])
        print(str(n + 32) + ":", output.split("\n")[-2].split(" ")[-1], end=" | ")
    print("")

    # p-e expl rerank    
    print("P-E Expl")
    for n in [0, 32, 64, 96]:
        output = run_cmd(args, ['python', 'calib_exps/calib_joint.py', '--style', JOINT_STYLE, '--sub_calib', str(n)])
        print(str(n + 32) + ":", output.split("\n")[-2].split(" ")[-1], end=" | ")
    print("")

if __name__=='__main__':
    args = _parse_args()
    for s in [0, 128, 256, 384, 512]:
        print(s)
        args.train_slice = s
        main_exp_group(args)

