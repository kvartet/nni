import nni
import copy
import argparse

from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats

def main(arch, args):
    for t in query_nb201_trial_stats(arch, 200, args.dataset, 'mean'):
        # pprint.pprint(t)
        print('acc = {}'.format(t["test_acc"]))
        print('flops = {}'.format(t["flops"]))
        nni.report_final_result(t["test_acc"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controller')
    parser.add_argument('--tuner', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    param = nni.get_next_parameter()
    arch = copy.deepcopy(param)
    print(arch)
    if args.tuner == 'tpe':
        arch = list(arch.values())[0]
    elif args.tuner == 'evolution':
        arch = list(arch.values())[0]
        del arch["_name"]
    elif args.tuner == 'ppo':
        print('do nothing')
    main(arch, args)
