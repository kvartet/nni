import pandas as pd
import time
import os
import argparse

from pathlib import Path
from nni.experiment import Experiment
from nni.algorithms.hpo.ppo_tuner.ppo_tuner import PPOTuner
from nni.algorithms.hpo.tpe import TPETuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner

parser = argparse.ArgumentParser(description='controller')
parser.add_argument('--main_file', type=str, default='trial_code_nni.py')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--search_space_path', type=str, default='search.json')
parser.add_argument('--flops', type=str, default='100')
parser.add_argument('--version', type=str, default='1')
parser.add_argument('--situation', type=str, default='aware')
parser.add_argument('--port', type=int, default=8741)
parser.add_argument('--tuner', type=str, default='tpe')
parser.add_argument('--dataset', type=str, default='cifar10') # imagenet16-120

args = parser.parse_args()
max_trial_number = 100

if args.tuner == 'ppo':
    tuner = PPOTuner(args.seed)
    max_trial_number = 1000
elif args.tuner == 'tpe':
    tuner = TPETuner(args.seed)
elif args.tuner == 'evolution':
    tuner = EvolutionTuner(seed=args.seed)

experiment = Experiment(tuner, 'local')

experiment.config.experiment_name = args.dataset + '_' + args.tuner + \
    '_'+ args.situation + '_' + args.flops + '_v' + args.version
experiment.config.max_experiment_duration = "4h"
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = max_trial_number
experiment.config.search_space_file = args.search_space_path
experiment.config.trial_command = 'python3 '+args.main_file
experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.gpu_indices = '1'
experiment.config.trial_gpu_number = 1
# experiment.config.tuner_gpu_indices='1'
experiment.start(args.port)

# print(experiment.get_experiment_profile()['params']['experimentName'])

while True:
    time.sleep(120)
    if experiment.get_status() == 'DONE':
        data_list = []
        for item in experiment.list_trial_jobs():
            if not item.finalMetricData:
                print(item)
                data_list.append(0)
            else:
                data_list.append(item.finalMetricData[0].data)

        path = 'results/' + args.dataset + '_' + args.tuner + '_' + args.situation + '.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
        else:
            df = pd.DataFrame()
        print(experiment.get_experiment_profile()['params']['experimentName'])
        df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
        df.to_csv(path, index=0, header=1)

        experiment.stop()
        break
