import nni
import pprint
from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats
import argparse
            
def get_all():
    keys = ['0_1', '0_2', '0_3', '1_2', '1_3', '2_3']
    values = ["none", "skip_connect", "avg_pool_3x3", "conv_1x1", "conv_3x3"]
    arch = {key: 'none' for key in keys}
    result_dict = {}
    num_dict = {}
    import time
    start = time.time()
    for v0 in values:
        arch[keys[0]] = v0
        for v1 in values:
            arch[keys[1]] = v1
            for v2 in values:
                arch[keys[2]] = v2
                for v3 in values:
                    arch[keys[3]] = v3
                    for v4 in values:
                        arch[keys[4]] = v4
                        for v5 in values:
                            arch[keys[5]] = v5
                            for t in query_nb201_trial_stats(arch, 200, 'cifar10','mean'):
                                if t["flops"] not in result_dict.keys() or t["test_acc"]>result_dict[t["flops"]]:
                                    result_dict[t["flops"]] = t["test_acc"]
                                if t["flops"] not in num_dict.keys():
                                    num_dict[t["flops"]] = 1
                                else:
                                    num_dict[t["flops"]] += 1
    end = time.time()
    print(end-start)
    print(result_dict)
    print('-----------------------')
    print(num_dict)




if __name__ == '__main__':
    get_all()