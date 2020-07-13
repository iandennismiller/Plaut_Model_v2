"""
simulator.py

=== SUMMARY ===
Description     : Load information from config file
Date Created    : July 12, 2020
Last Updated    : July 12, 2020

=== UPDATE NOTES ===
 > July 12, 2020
    - initial file creation
"""

import configparser
from datetime import datetime


class Config:

    def __init__(self, filepath):
        config = configparser.ConfigParser()
        config.read(filepath)

        self.Training = Config.Training(config)
        self.Checkpoint = Config.Checkpoint(config)
        self.Dataset = Config.Dataset(config)
        self.Optimizer = Config.Optimizer(config)
        self.General = Config.General(config, self.Dataset)

    class General:
        def __init__(self, config, Dataset):
            assert config['general']['label'] != '', "ERROR: Label must not be left blank"

            self.random_seed = int(config['general']['random_seed'])
            self.dilution = len(Dataset.anchor_sets)
            self.order = 1 if 1 in Dataset.anchor_sets else max(Dataset.anchor_sets)
            self.date = datetime.today().strftime("%Y_%m_%d")
            self.label = f"{config['general']['label']}-S{self.random_seed}D{self.dilution}O{self.order}-{self.date}"
            self.rootdir = None

    class Training:
        def __init__(self, config):
            self.plot_freq = int(config['training']['plot_freq'])
            self.print_freq = int(config['training']['print_freq'])
            self.save_freq = int(config['training']['save_freq'])
            self.target_radius = float(config['training']['target_radius'])
            self.total_epochs = int(config['training']['total_epochs'])
            self.anchor_epoch = int(config['training']['anchor_epoch'])

            assert self.plot_freq > 0 and self.print_freq > 0 and self.save_freq > 0, \
                "ERROR: Plot, Print, Save frequencies must be greater than 0"
            assert self.total_epochs > 0, "ERROR: Total Epochs must be greater than 0"
            assert self.anchor_epoch > 0, "ERROR: Anchor Epoch must be greater than 0"
            assert self.total_epochs >= self.anchor_epoch, \
                "ERROR: Total Epochs must be greater than or equal to Anchor Epoch"
            assert 0 <= self.target_radius <= 1, "ERROR: Target Radius must be between 0 and 1"

    class Checkpoint:
        def __init__(self, config):
            self.cp_epochs = [int(x) for x in config['checkpoint']['checkpoint_save_epochs'].split(',') if x != '']
            self.checkpoint_file = config['checkpoint']['checkpoint_file']

    class Dataset:
        def __init__(self, config):
            self.plaut_filepath = config['dataset']['plaut']
            self.anchor_filepath = config['dataset']['anchor']
            self.probe_filepath = config['dataset']['probe']
            self.anchor_sets = [int(x) for x in config['dataset']['anchor_sets'].split(',')]
            self.anchor_base_freq = float(config['dataset']['anc_freq'])
            self.plaut_types = [x.strip() for x in config['dataset']['track_plaut_types'].split(',')]
            self.anchor_types = [x.strip() for x in config['dataset']['track_anchor_types'].split(',')]
            self.probe_types = [x.strip() for x in config['dataset']['track_probe_types'].split(',')]

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
                except:
                    break

            assert set(self.optim_config['optimizer']).issubset({'Adam', 'SGD'}), "ERROR: Only Adam or SGD can be used."
            assert 1 in self.optim_config['start_epoch'], "ERROR: Must specify starting optimizer"
