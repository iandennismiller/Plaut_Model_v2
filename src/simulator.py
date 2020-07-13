"""
simulator.py

=== SUMMARY ===
Description     : Code for running simulation for training model and saving results
Date Created    : May 03, 2020
Last Updated    : July 12, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1
    - reorganization of config.cfg
        - allows user specification of word types to calculate accuracy for
    - method of calculating accuracy is changed:
        - the 'vowel only' option is kept, accuracy IS NOT be calculated over onset and coda graphemes
        - the use of pandas dataframe in calculations is replaced
        - to speed up training, accuracy is calculated concurrently with loss calculation
          (v1 required the DataLoader to be iterated twice)
        - option of saving a notes.txt after training is removed (it was rarely used in v1)
    - added option to delete folder if one with the same name exists
    - removed option to delete folder after error (this can be done with the option just above by rerunning the simulation again)
    - added assert statements for error checking of user input
    - anchors are now placed in one single csv file, and sets are chosen in config.cfg rather than making different csv files

=== UPDATE NOTES ===
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
    - modify anchor frequency to be user adjustible
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np

from src.dataset import Plaut_Dataset
from src.model import Plaut_Net
from src.helpers import *
from src.results_tool import Results

from config.config import Config


class Simulator():
    def __init__(self, config_filepath):
        """
        Initializes Simulator object

        Arguments:
            config_filepath {str} -- filepath of the configuration file
        """

        print("Initializing Simulator...")

        # load config and set random seed
        self.config = Config(config_filepath)
        torch.manual_seed(self.config.General.random_seed)
        print("--Configuration Loaded")

        # load data
        self.load_data()
        print("--Datasets Loaded")

        # create simulation folder
        rootdir, label = create_simulation_folder(self.config.General.label)
        self.config.General.label = label
        self.config.General.rootdir = rootdir
        print(f"--Simulation Results will be stored in: {self.config.General.rootdir}")

        # initialize model
        self.model = Plaut_Net()
        print("--Model Initialized")

    def load_data(self):
        """
        Creates the DataLoader objects for training
        """
        # create the custom datasets
        self.plaut_ds = Plaut_Dataset(self.config.Dataset.plaut_filepath)
        self.anchor_ds = Plaut_Dataset(self.config.Dataset.anchor_filepath)
        self.probe_ds = Plaut_Dataset(self.config.Dataset.probe_filepath)

        # get the types to track from the dataset if not given
        if self.config.Dataset.plaut_types == ['']:
            self.config.Dataset.plaut_types = list(self.plaut_ds.get_types())
        else:
            assert set(self.config.Dataset.plaut_types).issubset(
                self.plaut_ds.get_types()), "ERROR: Word types must exist in dataset."
        if self.config.Dataset.anchor_types == ['']:
            self.config.Dataset.anchor_types = list(self.anchor_ds.get_types())
        else:
            assert set(self.config.Dataset.anchor_types).issubset(
                self.anchor_ds.get_types()), "ERROR: Word types must exist in dataset."
        if self.config.Dataset.probe_types == ['']:
            self.config.Dataset.probe_types = list(self.probe_ds.get_types())
        else:
            assert set(self.config.Dataset.probe_types).issubset(
                self.probe_ds.get_types()), "ERROR: Word types must exist in dataset."

        # choose the specified anchor sets and set frequency appropriately
        self.anchor_ds.restrict_set(self.config.Dataset.anchor_sets)
        self.anchor_ds.set_frequency(
            self.config.Dataset.anchor_base_freq / len(self.config.Dataset.anchor_sets))

        self.plaut_samples = len(self.plaut_ds)
        self.anchor_samples = len(self.anchor_ds)
        self.probe_samples = len(self.probe_ds)

        # initialize DataLoaders
        self.plaut_loader = DataLoader(self.plaut_ds, batch_size=len(self.plaut_ds), num_workers=0)
        self.anchor_loader = DataLoader(self.anchor_ds, batch_size=len(self.anchor_ds), num_workers=0)
        self.probe_loader = DataLoader(self.probe_ds, batch_size=len(self.probe_ds), num_workers=0)

    def train(self):
        """
        Training Function for simulator
        """

        """ SETUP """
        # define loss function
        self.criterion = nn.BCELoss(reduction='none')

        # initialize results storage classes
        training_loss = Results(results_dir=self.config.General.rootdir + "/Training Loss",
                                config=self.config, title="Training Loss", labels=("Epoch", "Loss"))
        plaut_accuracy = Results(results_dir=self.config.General.rootdir + "/Training Accuracy", config=self.config,
                                 title="Training Accuracy", labels=("Epoch", "Accuracy"),
                                 categories=self.config.Dataset.plaut_types)
        anchor_accuracy = Results(results_dir=self.config.General.rootdir + "/Anchor Accuracy", config=self.config,
                                  title="Anchor Accuracy", labels=("Epoch", "Accuracy"),
                                  categories=self.config.Dataset.anchor_types)
        probe_accuracy = Results(results_dir=self.config.General.rootdir + "/Probe Accuracy", config=self.config,
                                 title="Probe Accuracy", labels=("Epoch", "Accuracy"),
                                 categories=self.config.Dataset.probe_types)
        output_data = Results(results_dir=self.config.General.rootdir, config=self.config,
                              title="Simulation Results", labels=('Epoch', ""),
                              categories=['example_id', 'orth', 'phon', 'category', 'correct', 'anchors_added'])
        time_data = Results(results_dir=self.config.General.rootdir, config=self.config,
                            title="Running Time", labels=("Epoch", "Time (s)"))
        hidden_layer_data = Results(results_dir=self.config.General.rootdir, config=self.config,
                                    title="Hidden Layer", categories=['orth', 'category', 'activation'])
        output_layer_data = Results(results_dir=self.config.General.rootdir, config=self.config,
                                    title="Output Layer", categories=['orth', 'category', 'activation'])

        start_epoch = 1

        """
        LOAD CHECKPOINT
        """
        if self.config.Checkpoint.checkpoint_file != "":
            start_epoch, optimizer = self.load_checkpoint()

        """ TRAINING LOOP """
        for epoch in range(start_epoch, self.config.Training.total_epochs + 1):
            epoch_time = time.time()
            epoch_loss = 0

            # change optimizer if needed
            if epoch in self.config.Optimizer.optim_config['start_epoch']:
                current_optim = self.config.Optimizer.optim_config['start_epoch'].index(epoch)
                optimizer = self.set_optimizer(current_optim)

            """ PLAUT DATASET """
            correct = np.zeros(len(self.config.Dataset.plaut_types))
            total = np.zeros(len(self.config.Dataset.plaut_types))
            compare, hl, ol = [], [], []

            data = None
            for i, data in enumerate(self.plaut_loader):
                loss, temp_correct, temp_total, temp_compare, temp_hl, temp_ol = self.predict(
                    data, categories=self.config.Dataset.plaut_types)  # find loss and accuracy

                epoch_loss += loss  # accumulate loss

                correct += temp_correct  # accumulate correct
                total += temp_total  # accumulate total
                compare += temp_compare
                hl.append(temp_hl)
                ol.append(temp_ol)

            # save plaut accuracy results
            plaut_accuracy.append_row(epoch, (correct / total).tolist())

            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1, 1 + self.plaut_samples)),
                'orth': data['orth'],
                'phon': data['phon'],
                'category': data['type'],
                'correct': compare,
                'anchors_added': [1 if epoch > self.config.Training.anchor_epoch else 0] * len(compare)})

            # save hidden and output layer data
            if epoch >= self.config.Training.anchor_epoch and epoch % self.config.Training.plot_freq == 0:
                hl = np.vstack(hl)  # concatenate data
                hidden_layer_data.add_rows([epoch] * hl.shape[0], {
                    'orth': data['orth'],
                    'category': data['type'],
                    'activation': hl.tolist()
                })
                ol = np.vstack(ol)
                output_layer_data.add_rows([epoch] * ol.shape[0], {
                    'orth': data['orth'],
                    'category': data['type'],
                    'activation': ol.tolist()
                })

            """ ANCHOR DATASET """
            correct = np.zeros(len(self.config.Dataset.anchor_types))
            total = np.zeros(len(self.config.Dataset.anchor_types))
            compare, hl, ol = [], [], []

            data = None
            for i, data in enumerate(self.anchor_loader):
                loss, temp_correct, temp_total, temp_compare, temp_hl, temp_ol = self.predict(
                    data, categories=self.config.Dataset.anchor_types)  # find loss and accuracy

                # accumulate loss when anchors are added into training set
                if epoch > self.config.Training.anchor_epoch:
                    epoch_loss += loss

                correct += temp_correct  # accumulate correct
                total += temp_total  # accumulate total
                compare += temp_compare
                hl.append(temp_hl)
                ol.append(temp_ol)

            # save anchor accuracy results
            anchor_accuracy.append_row(epoch, (correct / total).tolist())

            # save output data
            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1 + self.plaut_samples, 1 + self.plaut_samples + self.anchor_samples)),
                'orth': data['orth'],
                'phon': data['phon'],
                'category': data['type'],
                'correct': compare,
                'anchors_added': [1 if epoch > self.config.Training.anchor_epoch else 0] * len(compare)})

            # save hidden and output layer data
            if epoch >= self.config.Training.anchor_epoch and epoch % (self.config.Training.plot_freq / 10) == 0:
                hl = np.vstack(hl)  # concatenate data
                hidden_layer_data.add_rows([epoch] * hl.shape[0], {
                    'orth': data['orth'],
                    'category': data['type'],
                    'activation': hl.tolist()
                })
                ol = np.vstack(ol)
                output_layer_data.add_rows([epoch] * ol.shape[0], {
                    'orth': data['orth'],
                    'category': data['type'],
                    'activation': ol.tolist()
                })

            """ PROBE DATASET """
            correct = np.zeros(len(self.config.Dataset.probe_types))
            total = np.zeros(len(self.config.Dataset.probe_types))
            compare, hl, ol = [], [], []

            data = None
            for i, data in enumerate(self.probe_loader):
                loss, temp_correct, temp_total, temp_compare, temp_hl, temp_ol = self.predict(
                    data, categories=self.config.Dataset.probe_types)  # find loss and accuracy

                correct += temp_correct  # accumulate correct
                total += temp_total  # accumulate total
                compare += temp_compare
                hl.append(temp_hl)
                ol.append(temp_ol)

            # save probe accuracy results
            probe_accuracy.append_row(epoch, (correct / total).tolist())

            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1 + self.plaut_samples + self.anchor_samples,
                                         1 + self.plaut_samples + self.anchor_samples + self.probe_samples)),
                'orth': data['orth'],
                'phon': data['phon'],
                'category': data['type'],
                'correct': compare,
                'anchors_added': [1 if epoch > self.config.Training.anchor_epoch else 0] * len(compare)})

            # save hidden and output layer data
            if epoch >= self.config.Training.anchor_epoch and epoch % (self.config.Training.plot_freq / 10) == 0:
                hl = np.vstack(hl)
                hidden_layer_data.add_rows([epoch] * hl.shape[0], {
                    'orth': data['orth'],
                    'category': data['type'],
                    'activation': hl.tolist()
                })
                ol = np.vstack(ol)
                output_layer_data.add_rows([epoch] * ol.shape[0], {
                    'orth': data['orth'],
                    'category': data['type'],
                    'activation': ol.tolist()
                })

            """ UPDATE PARAMETERS, PLOT, SAVE """
            # calculate gradients and update weights
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save loss results
            training_loss.append_row(epoch, epoch_loss.item())

            # plot results
            if epoch % self.config.Training.plot_freq == 0:
                training_loss.lineplot()
                plaut_accuracy.lineplot()
                anchor_accuracy.lineplot()
                probe_accuracy.lineplot()

            # print statistics
            if epoch % self.config.Training.print_freq == 0:
                epoch_time = time.time() - epoch_time
                time_data.append_row(epoch, epoch_time)
                print(
                    f"[EPOCH {epoch}] \t loss: {epoch_loss.item():.4f} \t time: {epoch_time:.4f}")

            # save checkpoint
            if epoch in self.config.Checkpoint.cp_epochs:
                self.save_checkpoint(epoch, optimizer)

        """ SAVE RESULTS, PLOT, AND FINISH """
        # save output data
        total_samples = self.plaut_samples + self.anchor_samples + self.probe_samples
        output_data.add_columns({
            'error': [0] * len(output_data),
            'random_seed': [self.config.General.random_seed] * len(output_data),
            'dilution': [self.config.General.dilution] * len(output_data),
            'order': [self.config.General.order] * len(output_data)
        })

        for key in ['optimizer', 'learning_rate', 'momentum', 'weight_decay']:
            temp = []
            for i in range(len(self.config.Optimizer.optim_config['start_epoch'])):
                epoch_start = max(start_epoch, int(
                    self.config.Optimizer.optim_config['start_epoch'][i]))  # start of optimizer config
                # end of optimizer config is next item in start, or if no more items, then end is total epochs
                try:
                    epoch_end = min(int(
                        self.config.Optimizer.optim_config['start_epoch'][i + 1]),
                        self.config.Training.total_epochs + 1)
                except:
                    epoch_end = self.config.Training.total_epochs + 1
                for k in range(total_samples):  # once per every word
                    temp += [self.config.Optimizer.optim_config[key][i]
                             for j in range(epoch_start, epoch_end)]  # once per every epoch
            output_data.add_columns({key: temp})

        # save data as .csv.gz files and produce final plots
        output_data.save_data(index_label='epoch')
        hidden_layer_data.save_data(index_label='epoch')
        output_layer_data.save_data(index_label='epoch')
        time_data.lineplot()
        plaut_accuracy.barplot()
        anchor_accuracy.barplot()
        probe_accuracy.barplot()

    def set_optimizer(self, i):
        """
        Changes the optimizer based on configuration settings

        Arguments:
            i {int} -- optimizer index

        Returns:
            torch.optim.x.x -- the specified optimizer
        """
        if self.config.Optimizer.optim_config['optimizer'][i] == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.config.Optimizer.optim_config['learning_rate'][i],
                              weight_decay=self.config.Optimizer.optim_config['weight_decay'][i])
        else:
            return optim.SGD(self.model.parameters(), lr=self.config.Optimizer.optim_config['learning_rate'][i],
                             weight_decay=self.config.Optimizer.optim_config['weight_decay'][i],
                             momentum=self.config.Optimizer.optim_config['momentum'][i])

    def predict(self, data, categories=None):
        """
        Makes predictions given the current model, as well as calculate loss and accuracy
        (Accuracy given as two seperate lists of correct and total)

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
        hl, outputs = self.model(inputs)

        # target radius
        if self.config.Training.target_radius > 0:
            target_one_indices = torch.where(targets == 1)
            target_zero_indices = torch.where(targets == 0)
            target_upper_thresh = torch.full(targets.shape, 1 - self.config.Training.target_radius)
            target_lower_thresh = torch.full(targets.shape, self.config.Training.target_radius)
            targets[target_one_indices] = torch.max(target_upper_thresh, outputs.detach())[target_one_indices]
            targets[target_zero_indices] = torch.min(target_lower_thresh, outputs.detach())[target_zero_indices]

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

        return loss, np.array(correct), np.array(total), compare, hl.detach().numpy(), outputs.detach().numpy()

    def save_checkpoint(self, epoch, optimizer):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_name': optimizer.__class__.__name__
        }, f"checkpoints/{self.config.General.label}_{epoch}.tar")

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.Checkpoint.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        assert {'model_state_dict', 'optimizer_state_dict', 'epoch',
                'optimizer_name'}.issubset(set(checkpoint.keys()))

        if checkpoint['optimizer_name'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        else:
            optimizer = optim.Adam(self.model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'] + 1, optimizer


"""
TESTING AREA
"""
if __name__ == '__main__':
    sim = Simulator("config/config.cfg")
    sim.train()
