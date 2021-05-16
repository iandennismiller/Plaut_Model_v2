"""
simulator.py

=== SUMMARY ===
Description     : Code for running simulation for training model and saving results
Date Created    : May 03, 2020
Last Updated    : September 20, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1
    - reorganization of simulator_config.cfg
        - allows user specification of word types to calculate accuracy for
    - method of calculating accuracy is changed:
        - the 'vowel only' option is kept, accuracy IS NOT be calculated over onset and coda graphemes
        - the use of pandas dataframe in calculations is replaced
        - to speed up training, accuracy is calculated concurrently with loss calculation
          (v1 required the DataLoader to be iterated twice)
        - option of saving a notes.txt after training is removed (it was rarely used in v1)
    - added option to delete folder if one with the same name exists
    - removed option to delete folder after error (this can be done with the option above by rerunning the simulation)
    - added assert statements for error checking of user input
    - anchors are now placed in one single csv file, and anchor sets are chosen in simulator_config.cfg

=== UPDATE NOTES ===
 > September 20, 2020
    - migrate results objects to results_handler class
    - update function for saving checkpoints
    - update function for setting optimizer based on simulator config changes
 > September 7, 2020
    - remove logging for creating checkpoints
 > August 30, 2020
    - remove function for loading checkpoints
 > July 27, 2020
    - save hidden layer and output layer data starting from epoch 0
 > July 26, 2020
    - implement "generalized" cross entropy loss
    - add series folder
 > July 19, 2020
    - remove DataLoaders, simply use the existing Dataset class
    - minor logging change
    - incorporate changes from adding constants classes
    - formatting changes
 > July 18, 2020
    - minor reformatting changes
    - add preliminary logging system
 > July 12, 2020
    - add target radius in predict function
 > June 13, 2020
    - update to save output layer data as well
 > May 24, 2020
    - update string format, filepath, import statements
    - code reformat and reorganization
    - add saving of hidden layer data
 > May 08, 2020
    - add saving and loading checkpoints
    - add accuracy bar plots at end of training
 > May 07, 2020
    - Edit method of loading parameters from config file
    - add saving of output data in csv
    - modify anchor frequency to be user adjustable
    - adjust code based on changes in the results class
    - add additional docstrings for functions
 > May 06, 2020
    - adjust code based on changes in the results class
 > May 05, 2020
    - add accuracy computations, plotting, change epochs to start from 1
    - limit decimal places in loss/time printout
 > May 03, 2020
    - file created, basic simulator created
    - add assert statements for error checking of user input

"""

import logging
import time
from tqdm import tqdm
import copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from config.simulator_config import Config
from common.helpers import *
from common.constants import WordTypes
from simulator.dataset import PlautDataset
from simulator.model import PlautNet
from simulator.results import Results
from simulator.results_handler import ResultsHandler
from simulator.GCELoss import GCELoss


class Simulator:
    def __init__(self, config_filepath, series=False):
        """
        Initializes Simulator object

        Arguments:
            config_filepath {str} -- filepath of the configuration file
        """
        # initialize logger
        self.logger = logging.getLogger('__main__.' + __name__)
        self.logger.info("Initializing Simulator")

        # load config and set random seed
        self.config = Config(config_filepath)
        torch.manual_seed(self.config.General.random_seed)
        self.logger.debug("Configuration successfully loaded")

        # load datasets
        self.plaut_ds, self.anchor_ds, self.probe_ds = self.load_data()
        self.plaut_size, self.anchor_size, self.probe_size = len(self.plaut_ds), len(self.anchor_ds), len(self.probe_ds)
        self.logger.debug("Datasets successfully loaded")

        # create simulation folder
        rootdir, label = create_simulation_folder(self.config.General.label, series)
        self.config.General.label = label
        self.config.General.rootdir = rootdir
        self.logger.info(f"Simulation Results will be stored in: {self.config.General.rootdir}")

        # initialize model
        self.model = PlautNet()
        self.criterion = None
        self.optimizer = None
        self.epoch = -1
        self.logger.debug("Model successfully initialized")
        self.logger.info("Simulator initialization completed")

    def load_data(self):
        """
        Creates the DataLoader objects for training
        """
        # create the custom datasets
        plaut_ds = PlautDataset(self.config.Dataset.plaut_filepath)
        anchor_ds = PlautDataset(self.config.Dataset.anchor_filepath)
        probe_ds = PlautDataset(self.config.Dataset.probe_filepath)

        # choose the specified anchor sets and set frequency appropriately
        anchor_ds.restrict_set(self.config.Dataset.anchor_sets)
        anchor_ds.set_frequency(self.config.Dataset.anchor_base_freq / len(self.config.Dataset.anchor_sets))

        # return datasets and sizes
        return plaut_ds, anchor_ds, probe_ds

    def train(self):
        """
        Training Function for simulator
        """

        """ SETUP """
        # define loss function (generalized cross entropy loss)
        self.criterion = nn.BCELoss(reduction='none')

        # initialize results storage classes
        results_handler = ResultsHandler(config=self.config)

        output_data = Results(results_dir=self.config.General.rootdir,
                              config=self.config,
                              title="Simulation Results",
                              labels=('Epoch', ""),
                              columns=['example_id', 'orth', 'phon', 'category', 'correct', 'anchors_added'])

        """ TRAINING LOOP """
        t = tqdm(range(1, self.config.Training.total_epochs + 1), smoothing=0.15)
        for epoch in t:
            _epoch_time = time.time()
            self.epoch = epoch
            epoch_loss = 0

            # change optimizer if needed
            self.set_optimizer()

            """ PLAUT DATASET """
            data = self.plaut_ds[:]  # load data
            loss, accuracy, activations = self.predict(data, categories=WordTypes.plaut_types)
            correct, total, compare = accuracy

            epoch_loss += loss  # accumulate loss

            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1, 1 + self.plaut_size)),
                'orth': data['orth'],
                'phon': data['phon'],
                'category': data['type'],
                'correct': compare,
                'anchors_added': [1 if epoch > self.config.Training.anchor_epoch else 0] * len(compare)})

            # save data
            results_handler.add_data(category='plaut_accuracy', epoch=self.epoch, data=(correct / total).tolist())
            results_handler.add_data(category='activations', epoch=self.epoch, data=('plaut', data, activations))

            """ ANCHOR DATASET """
            data = self.anchor_ds[:]  # load data
            loss, accuracy, activations = self.predict(data, categories=WordTypes.anchor_types)
            correct, total, compare = accuracy

            # accumulate loss when anchors are added into training set
            if epoch > self.config.Training.anchor_epoch:
                epoch_loss += loss

            # save output data
            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1 + self.plaut_size, 1 + self.plaut_size + self.anchor_size)),
                'orth': data['orth'],
                'phon': data['phon'],
                'category': data['type'],
                'correct': compare,
                'anchors_added': [1 if epoch > self.config.Training.anchor_epoch else 0] * len(compare)})

            # save data
            results_handler.add_data(category='anchor_accuracy', epoch=self.epoch, data=(correct / total).tolist())
            results_handler.add_data(category='activations', epoch=self.epoch, data=('anchor', data, activations))

            """ PROBE DATASET """
            data = self.probe_ds[:]
            loss, accuracy, activations = self.predict(data, categories=WordTypes.probe_types)  # find loss and accuracy
            correct, total, compare = accuracy

            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1 + self.plaut_size + self.anchor_size,
                                         1 + self.plaut_size + self.anchor_size + self.probe_size)),
                'orth': data['orth'],
                'phon': data['phon'],
                'category': data['type'],
                'correct': compare,
                'anchors_added': [1 if epoch > self.config.Training.anchor_epoch else 0] * len(compare)})

            # save hidden and output layer data
            results_handler.add_data(category='probe_accuracy', epoch=self.epoch, data=(correct / total).tolist())
            results_handler.add_data(category='activations', epoch=self.epoch, data=('probe', data, activations))

            """ UPDATE PARAMETERS, PLOT, SAVE """
            # calculate gradients and update weights
            epoch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # save loss results and model weights
            results_handler.add_data(category='loss', epoch=self.epoch, data=epoch_loss.item())
            results_handler.add_data(category='weights', epoch=self.epoch, data=copy.deepcopy(self.model.state_dict()))

            # plot results
            results_handler.create_training_plots(epoch=self.epoch)

            # save checkpoint
            self.save_checkpoint()

            # calculate epoch time
            _epoch_time = time.time() - _epoch_time
            results_handler.add_data(category='running_time', epoch=self.epoch, data=_epoch_time)

            t.set_description(f"[Epoch {epoch:3d}] loss: {epoch_loss.item():9.2f} | time: {_epoch_time:.4f} |\tProgress")

        """ SAVE RESULTS, PLOT, AND FINISH """
        # save output data
        total_samples = self.plaut_size + self.anchor_size + self.probe_size
        output_data.add_columns({
            'error': [0] * len(output_data),
            'random_seed': [self.config.General.random_seed] * len(output_data),
            'dilution': [self.config.General.dilution] * len(output_data),
            'order': [self.config.General.order] * len(output_data)
        })

        for key in ['optimizer', 'learning_rate', 'momentum', 'weight_decay']:
            temp = []
            start_epochs = sorted(list(self.config.Optimizer.optim_config.keys()))
            for i in range(len(start_epochs)):
                epoch_start = start_epochs[i]  # start of optimizer config
                # end of optimizer config is next item in start, or if no more items, then end is total epochs
                try:
                    epoch_end = min(start_epochs[i+1], self.config.Training.total_epochs + 1)
                except IndexError:
                    epoch_end = self.config.Training.total_epochs + 1
                for k in range(total_samples):  # once per every word
                    temp += [self.config.Optimizer.optim_config[epoch_start][key]] * (epoch_end - epoch_start)
            output_data.add_columns({key: temp})

        # save data as .csv.gz files and produce final plots
        # output_data.save_data(index_label='epoch')
        results_handler.save_data()
        results_handler.create_final_plots()

        self.logger.info('Simulation completed.')

    def set_optimizer(self):
        """
        Changes the optimizer based on configuration settings

        Arguments:
            None

        Returns:
            None
        """
        if self.epoch in self.config.Optimizer.optim_config.keys():
            current_optim = self.config.Optimizer.optim_config[self.epoch]

            if current_optim['optimizer'] == 'Adam':
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=current_optim['learning_rate'],
                                            weight_decay=current_optim['weight_decay'])
            else:
                self.optimizer = optim.SGD(self.model.parameters(),
                                           lr=current_optim['learning_rate'],
                                           momentum=current_optim['momentum'],
                                           weight_decay=current_optim['weight_decay'])

    def predict(self, data, categories=None):
        """
        Makes predictions given the current model, as well as calculate loss and accuracy
        (Accuracy given as two separate lists of correct and total)

        Arguments:
            data {dict} -- dictionary of data given by the DataLoader

        Keyword Arguments:
            categories {list} -- list of word categories to calculate accuracy over (default: None)

        Returns:
            float -- loss
            np.array -- number of correct words per category
            np.array -- total number of words per category
            list -- True/False representing correctness of each word
        """

        if categories is None:
            categories = ['All']

        # extract log frequencies, grapheme vectors, phoneme vectors
        log_freq = data['log_freq'].view(-1, 1)
        inputs = data['graphemes']
        targets = data['phonemes'].clone()

        # forward pass
        hl_outputs, outputs = self.model(inputs)
        # clip outputs to prevent division by zero in loss calculation
        outputs_clipped = torch.min(outputs, torch.full(outputs.shape, 1-self.config.Training.target_radius))

        # target radius
        if self.config.Training.target_radius > 0:
            target_one_indices = torch.where(targets == 1)
            target_zero_indices = torch.where(targets == 0)
            target_upper_thresh = torch.full(targets.shape, 1 - self.config.Training.target_radius)
            target_lower_thresh = torch.full(targets.shape, self.config.Training.target_radius)
            targets[target_one_indices] = torch.max(target_upper_thresh, outputs_clipped.detach())[target_one_indices]
            targets[target_zero_indices] = torch.min(target_lower_thresh, outputs_clipped.detach())[target_zero_indices]

            loss = self.criterion(outputs_clipped, targets)
        else:
            loss = self.criterion(outputs, targets)

        # find weighted sum of loss
        loss = loss * log_freq
        loss = loss.sum()

        # accuracy calculations - accuracy is based on highest activity vowel phoneme
        outputs_max_vowel = outputs[:, 23:37].argmax(dim=1)
        targets_max_vowel = targets[:, 23:37].argmax(dim=1)
        compare = torch.eq(outputs_max_vowel, targets_max_vowel).tolist()

        correct, total = [], []
        # find correct and total for each category
        for cat in categories:
            if cat == 'All':  # find accuracy across all categories
                correct.append(sum(compare) / len(data['type']))
                total.append(len(data['type']))
            else:
                temp_total, temp_correct = 0, 0
                for t, c in zip(data['type'], compare):
                    if t == cat:  # if same category
                        temp_total += 1
                        if c:  # if correct
                            temp_correct += 1
                total.append(temp_total)
                correct.append(temp_correct)

        return loss, (np.array(correct), np.array(total), compare), \
            (hl_outputs.detach().numpy(), outputs.detach().numpy())

    def save_checkpoint(self):
        """
        Saves a checkpoint of the model weights

        Arguments:
            None

        Returns:
            None
        """
        if self.config.Checkpoint.save_epochs == [] and self.config.Checkpoint.save_frequency is None:
            return None
        if self.epoch in self.config.Checkpoint.save_epochs or self.epoch % self.config.Checkpoint.save_frequency == 0:
            torch.save(self.model.state_dict(), f'checkpoints/{self.config.General.label}_{self.epoch}.tar')


"""
TESTING AREA
"""
if __name__ == '__main__':
    sim = Simulator("config/simulator_config.cfg")
    sim.train()
