from pathlib import Path
from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.experiment_name = 'nasbench201'
experiment.config.max_experiment_duration = "0.5h"
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10
# experiment.config.search_space = search_space
experiment.config.trial_code_directory = '/home/v-yiruxu/nni/test/flops_test/'
experiment.config.search_space_file = ('/home/v-yiruxu/nni/test/flops_test/search.json')
experiment.config.trial_command = 'python3 trial_code.py'

experiment.config.training_service.use_active_gpu = False
experiment.config.tuner.name = 'TPE'

experiment.start(8475)
