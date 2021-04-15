import nni

from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats

def main(arch):
    for t in query_nb201_trial_stats(arch, 200, 'cifar100','mean'):
        # pprint.pprint(t)
        print(t["test_acc"])
        nni.report_final_result(t["test_acc"])

if __name__ == '__main__':
    arch = nni.get_next_parameter()
    print(arch)
    arch = list(arch.values())[0]
    del arch["_name"]
    main(arch)
