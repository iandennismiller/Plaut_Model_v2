"""
simulator_config.py

=== SUMMARY ===
Description     : Load information from config file
Date Created    : July 12, 2020
Last Updated    : September 8, 2020

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
import logging


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
            self.save_frequency = config['checkpoint']['save_frequency']

            if not self.save_epochs:
                self.save_epochs = []

            assert not all([self.save_epochs, self.save_frequency]), "ERROR: Either save epochs or save frequency " \
                                                                     "can be specified, but not both."

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
            self.sim_results = config['outputs']['sim_results']
            self.weights = config['outputs']['weights']

    class Optimizer:
        def __init__(self, config):
            self.optim_config = config['optimizers']
            assert 1 in self.optim_config.keys(), "ERROR: Starting optimizer (at epoch 1) must be specified"
            for key, value in self.optim_config.items():
                assert key > 0, "ERROR: Start epoch of optimizer must be greater than 0"
                assert value['optimizer'] in ['SGD', 'Adam'], "ERROR: Only SGD or Adam optimizers are supported"
                assert value['learning_rate'] > 0, "ERROR: Learning rate must be greater than 0"
                assert value['momentum'] >= 0 if value['optimizer'] == 'SGD' else True, "ERROR: Momentum must be " \
                                                                                        "greater than or equal to 0"
                assert value['weight_decay'] >= 0, "ERROR: Weight decay must be greater than 0"

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
