import pandas as pd
import time

from pathlib import Path
from nni.algorithms.hpo.tpe import TPETuner
from nni.experiment import Experiment

aware = False
tuner = TPETuner(aware=aware)
experiment = Experiment(tuner, 'local')
if aware:
    experiment.config.experiment_name = 'cifar100_tpe_aware_50_v3'
else:
    experiment.config.experiment_name = 'cifar100_tpe_noaware_50_v3'
experiment.config.max_experiment_duration = "0.5h"
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 100
# experiment.config.search_space = search_space
experiment.config.search_space_file = ('search.json')
experiment.config.trial_command = 'python3 trial_code.py'
# experiment.config.experiment_working_directory = '~/nni-experiments/' 


experiment.start(8475)

while True:
    time.sleep(120)
    if experiment.get_status() == 'DONE':
        data_list = []
        for item in experiment.get_job_metrics().values():
            data_list.append(item[0].data)

        # df = pd.DataFrame()
        if aware:
            df = pd.read_csv('results/tpe_aware.csv', low_memory=False)
            df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
            df.to_csv('results/tpe_aware.csv', index=0, header=1)
        else:
            df = pd.read_csv('results/tpe_noaware.csv', low_memory=False)
            df[experiment.get_experiment_profile()['params']['experimentName']] = data_list
            df.to_csv('results/tpe_noaware.csv', index=0, header=1)
        
        experiment.stop()
        break
