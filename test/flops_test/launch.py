import pandas as pd
import time

from pathlib import Path
from nni.algorithms.hpo.tpe import TPETuner
from nni.experiment import Experiment
import argparse
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner

parser = argparse.ArgumentParser(description='controller')
parser.add_argument('--main_file', type=str, default='trial_code.py')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--search_space_path', type=str, default='search.json')
parser.add_argument('--flops', type=str, default='100')
parser.add_argument('--version', type=str, default='1')
parser.add_argument('--situation', type=str, default='aware')

args = parser.parse_args()


tuner = EvolutionTuner(seed=args.seed)
experiment = Experiment(tuner, 'local')

experiment.config.experiment_name = 'cifar100_evolution_'+args.situation+'_'+args.flops+'_v'+args.version
experiment.config.max_experiment_duration = "0.5h"
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 100
# experiment.config.search_space = search_space
experiment.config.search_space_file = (args.search_space_path)
experiment.config.trial_command = 'python3 '+args.main_file
# experiment.config.experiment_working_directory = '~/nni-experiments/' 

experiment.start(8741)

while True:
    time.sleep(120)
    if experiment.get_status() == 'DONE':
        data_list = []
        for item in experiment.get_job_metrics().values():
            data_list.append(item[0].data)

        # df = pd.DataFrame()
        if args.situation == 'aware':
            df = pd.read_csv('results/evolution_aware.csv', low_memory=False)
            df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
            df.to_csv('results/evolution_aware.csv', index=0, header=1)
        elif args.situation == 'nonested':
            df = pd.read_csv('results/evolution_nonested.csv', low_memory=False)
            df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
            df.to_csv('results/evolution_nonested.csv', index=0, header=1)
        elif args.situation == 'noaware':
            df = pd.read_csv('results/evolution_noaware.csv', low_memory=False)
            df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
            df.to_csv('results/evolution_noaware.csv', index=0, header=1)
        
        experiment.stop()
        break




# aware = False
# tuner = TPETuner(aware=aware)
# experiment = Experiment(tuner, 'local')
# if aware:
#     experiment.config.experiment_name = 'cifar100_tpe_aware_50_v3'
# else:
#     experiment.config.experiment_name = 'cifar100_tpe_noaware_50_v3'
# experiment.config.max_experiment_duration = "0.5h"
# experiment.config.trial_concurrency = 1
# experiment.config.max_trial_number = 100
# # experiment.config.search_space = search_space
# experiment.config.search_space_file = ('search.json')
# experiment.config.trial_command = 'python3 trial_code.py'
# # experiment.config.experiment_working_directory = '~/nni-experiments/' 


# experiment.start(8475)

# while True:
#     time.sleep(120)
#     if experiment.get_status() == 'DONE':
#         data_list = []
#         for item in experiment.get_job_metrics().values():
#             data_list.append(item[0].data)

#         # df = pd.DataFrame()
#         if aware:
#             df = pd.read_csv('results/tpe_aware.csv', low_memory=False)
#             df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
#             df.to_csv('results/tpe_aware.csv', index=0, header=1)
#         else:
#             df = pd.read_csv('results/tpe_noaware.csv', low_memory=False)
#             df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
#             df.to_csv('results/tpe_noaware.csv', index=0, header=1)
        
#         experiment.stop()
#         break
