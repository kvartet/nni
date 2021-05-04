import os
import torch
import random

import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl

from dataset import TitanicDataset
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment


class Net(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.hidden_dim1 = nn.ValueChoice(
            [16, 32, 64, 128, 256, 512, 1024], label='hidden_dim1')
        self.hidden_dim2 = nn.ValueChoice(
            [16, 32, 64, 128, 256, 512, 1024], label='hidden_dim2')

        self.fc1 = nn.Linear(input_size, self.hidden_dim1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim1)
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.0, 0.25, 0.5]))

        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2)
        self.dropout2 = nn.Dropout(nn.ValueChoice([0.0, 0.25, 0.5]))

        self.fc3 = nn.Linear(self.hidden_dim2, 2)

    def forward(self, x):

        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))
        return x

# Step 1: Prepare the dataset
train_dataset = TitanicDataset('./data', train=True)
test_dataset = TitanicDataset('./data', train=False)

# Step 2: Define the Model Space
model_space = Net(len(train_dataset.__getitem__(0)[0]))

# Step 3: Explore the Defined Model Space

# Step 3.1: Choose a Search Strategy
# more startegies refer to https://nni.readthedocs.io/en/latest/NAS/retiarii/ApiReference.html#strategies
simple_strategy = strategy.TPEStrategy()

# Step 3.2: Choose or Write a Model Evaluator
trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=16),
                            val_dataloaders=pl.DataLoader(
    test_dataset, batch_size=16),
    max_epochs=20)

# Step 4: Configure the Experiment
exp = RetiariiExperiment(model_space, trainer, [], simple_strategy)

exp_config = RetiariiExeConfig('aml')
exp_config.experiment_name = 'titanic_example'
exp_config.trial_concurrency = 2
exp_config.max_trial_number = 20
exp_config.max_experiment_duration = '2h'
exp_config.trial_gpu_number = 1
exp_config.nni_manager_ip = '' # your nni_manager_ip

# training service config
exp_config.training_service.use_active_gpu = True
exp_config.training_service.subscription_id = '' # your subscription id
exp_config.training_service.resource_group = '' # your resource group
exp_config.training_service.workspace_name = '' # your workspace name
exp_config.training_service.compute_target = '' # your compute target
exp_config.training_service.docker_image = ''  # your docker image

# Step 5: Run and View the Experiment
exp.run(exp_config, 8081 + random.randint(0, 100))

# Step 6: Export the top Model
print('Final model:')
for model_code in exp.export_top_models():
    print(model_code)
