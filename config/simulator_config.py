"""
simulator_config.py

=== SUMMARY ===
Description     : Load information from config file
Date Created    : July 12, 2020
Last Updated    : July 18, 2020

=== UPDATE NOTES ===
 > September 7, 2020
    - minor fix to assert statement
 > August 30, 2020
    - removal of checkpoint filepath
 > July 19, 2020
    - remove option of specifying certain word types to track
 > July 18, 2020
    - date format change
    - fix broad except clause
 > July 12, 2020
    - initial file creation
"""

import os
import yaml
from datetime import datetime


class Config:
    def __init__(self, filepath):
        with open(filepath) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.Training = Config.Training(config)
        self.Checkpoint = Config.Checkpoint(config)
        self.Dataset = Config.Dataset(config)
        self.Outputs = Config.Outputs(config)
        self.Optimizer = Config.Optimizer(config)
        self.General = Config.General(config, self.Dataset)

    class Training:
        def __init__(self, config):
            self.anchor_epoch = int(config['training']['anchor_epoch'])
            self.total_epochs = int(config['training']['total_epochs'])
            self.target_radius = config['training']['target_radius']

            # ERROR CHECKING
            assert self.anchor_epoch > 0, "ERROR: Anchor Epoch must be greater than 0"
            assert self.total_epochs >= self.anchor_epoch, "ERROR: Total Epochs must be greater than or " \
                                                           "equal to Anchor Epoch"
            assert 0 <= self.target_radius <= 1, "ERROR: Target Radius must be between 0 and 1"

    class Checkpoint:
        def __init__(self, config):
            self.save_epochs = config['checkpoint']['save_epochs']
            if not self.save_epochs:
                self.save_epochs = []

    class Dataset:
        def __init__(self, config):
            self.plaut_filepath = config['dataset']['plaut']
            self.anchor_filepath = config['dataset']['anchor']
            self.probe_filepath = config['dataset']['probe']
            self.anchor_sets = config['dataset']['anchor_sets']
            self.anchor_base_freq = config['dataset']['anchor_freq']

            # ERROR CHECKING
            for fp in [self.plaut_filepath, self.anchor_filepath, self.probe_filepath]:
                assert os.path.isfile(fp), f"{fp} does not exist"

    class Outputs:
        def __init__(self, config):
            self.plotting = config['outputs']['plotting']
            self.hidden_activations = config['outputs']['activations']['hidden']
            self.output_activations = config['outputs']['activations']['output']
            self.weights = config['outputs']['weights']

    class Optimizer:
        def __init__(self, config):
            self.optim_config = {
                'start_epoch': [],
                'optimizer': [],
                'learning_rate': [],
                'momentum': [],
                'weight_decay': []
            }
            i = 1
            while True:
                try:
                    self.optim_config['start_epoch'].append(int(config['optim' + str(i)]['start_epoch']))
                    self.optim_config['optimizer'].append(config['optim' + str(i)]['optimizer'])
                    for category in ['learning_rate', 'momentum', 'weight_decay']:
                        self.optim_config[category].append(float(config['optim' + str(i)][category]))
                    i += 1
                except KeyError:
                    break

            assert set(self.optim_config['optimizer']).issubset({'Adam', 'SGD'}), "ERROR: Only Adam or SGD can be used."
            assert 1 in self.optim_config['start_epoch'], "ERROR: Must specify starting optimizer"

    class General:
        def __init__(self, config, Dataset):
            assert config['general']['label'] != '', "ERROR: Label must not be left blank"

            self.random_seed = config['general']['random_seed']
            self.dilution = len(Dataset.anchor_sets)
            self.order = 1 if 1 in Dataset.anchor_sets else max(Dataset.anchor_sets)
            self.date = datetime.today().strftime("%Y%m%d")
            self.label = f"{config['general']['label']}-S{self.random_seed}D{self.dilution}O{self.order}-{self.date}"
            self.rootdir = None


if __name__ == "__main__":
    Config('simulator_config.yaml')
