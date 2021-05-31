import pprint
import time

from nni.nas.benchmarks.nasbench101 import query_nb101_trial_stats
from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats
from nni.nas.benchmarks.nds import query_nds_trial_stats

ti = time.time()


arch = {'0_1': 'avg_pool_3x3', '0_2': 'avg_pool_3x3', '0_3': 'none', '1_2': 'avg_pool_3x3', '1_3': 'conv_3x3'}



for t in query_nb201_trial_stats(arch, 200, 'cifar100'):
    pprint.pprint(t)
