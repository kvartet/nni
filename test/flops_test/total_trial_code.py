import nni
import pprint
from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats
import argparse

def main(arch):
    parser = argparse.ArgumentParser(description='controller')
    parser.add_argument('--flops', type=int, default='0')
    args = parser.parse_args()
    
    for t in query_nb201_trial_stats(arch, 200, 'cifar100','mean'):
        # nni.report_final_result(t["test_acc"])
        # pprint.pprint(t)
        # print(t)
        if t["flops"] > args.flops:
            nni.report_final_result(0.001)
            print('acc = 0')
            print('flops = {}'.format(t["flops"]))
        else:
            nni.report_final_result(t["test_acc"])
            print('acc = {}'.format(t["test_acc"]))
            print('flops = {}'.format(t["flops"]))
            

if __name__ == '__main__':
    arch = nni.get_next_parameter()
    print(arch)
    main(arch)