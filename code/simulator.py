"""
simulator.py

=== SUMMARY ===
Description     : Code for running simulation for training model and saving results
Date Created    : May 03, 2020
Last Updated    : May 03, 2020

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
import datetime
import numpy as np
import configparser

from dataset import Plaut_Dataset
from model import Plaut_Net
from helpers import *
from results_tool import Results



class Simulator():
    def __init__(self, config_filepath):
        """
        Initializes Simulator object

        Arguments:
            config_filepath {str} -- filepath of the configuration file
        """

        print("Initializing Simulator...")

        self.configure_simulation(config_filepath) # load configuration
        print("--Configuration Loaded")
        self.load_data() # load data
        print("--Datasets Loaded")
        self.rootdir = create_simulation_folder(self.label) # create simulation folder
        print("--Simulation Results will be stored in: {}".format(self.rootdir))
        self.model = Plaut_Net() # initialize model
        print("--Model Initialized")
    
    def configure_simulation(self, filepath):
        """
        Loads configuration file, and sets parameters

        Detailed Description:
         - loads all configuration settings
         - sets random seed
         - sets simulation label

        Arguments:
            filepath {str} -- filepath of the configuration file
        """
        config = configparser.ConfigParser()
        config.read(filepath)
        
        # General Settings
        self.label = config['general']['label']
        
        # Training Settings
        for param in config['training']:
            exec(f"self.{param} = {int(config['training'][param])}")

        assert self.plot_freq > 0 and self.print_freq > 0 and self.save_freq > 0, "ERROR: Plot, Print, Save frequencies must be greater than 0"
        assert self.label != '', "ERROR: Label must not be left blank"
        assert self.total_epochs > 0, "ERROR: Total Epochs must be greater than 0"
        assert self.anchor_epoch > 0, "ERROR: Anchor Epoch must be greater than 0"
        assert self.total_epochs >= self.anchor_epoch, "ERROR: Total Epochs must be greater than or equal to Anchor Epoch"
        
        # Checkpoint Settings
        self.cp_epochs = [int(x) for x in config['checkpoint']['checkpoint_epochs'].split(',')]
        self.cp_name = config['checkpoint']['checkpoint_name']
        self.prev_checkpoint = config['checkpoint']['prev_checkpoint']

        # Dataset Settings
        self.plaut_filepath = config['dataset']['plaut']
        self.anchor_filepath = config['dataset']['anchor']
        self.probe_filepath = config['dataset']['probe']
        self.anchor_sets = [int(x) for x in config['dataset']['anchor_sets'].split(',')]
        self.anchor_base_freq = float(config['dataset']['anc_freq'])
        self.plaut_types = [x.strip() for x in config['dataset']['track_plaut_types'].split(',')]
        self.anchor_types = [x.strip() for x in config['dataset']['track_anchor_types'].split(',')]
        self.probe_types = [x.strip() for x in config['dataset']['track_probe_types'].split(',')]

        # Optimizer Settings
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
                self.optim_config['start_epoch'].append(int(config['optim'+str(i)]['start_epoch']))
                self.optim_config['optimizer'].append(config['optim'+str(i)]['optimizer'])
                for category in ['learning_rate', 'momentum', 'weight_decay']:
                    self.optim_config[category].append(float(config['optim'+str(i)][category]))
                i += 1
            except:
                break
        
        assert set(self.optim_config['optimizer']).issubset({'Adam', 'SGD'}), "ERROR: Only Adam or SGD can be used."
        assert 1 in self.optim_config['start_epoch'], "ERROR: Must specify starting optimizer"
        
        self.current_optim = 0
        
        # set random seed
        torch.manual_seed(self.random_seed)

        # find date
        now = datetime.datetime.now()
        self.date = now.strftime("%b").lower()+now.strftime("%d")
        
        self.dilution = len(self.anchor_sets)
        self.order = 1 if 1 in self.anchor_sets else 3

        # set simulation label
        self.label += "-S{}D{}O{}-{}".format(self.random_seed, self.dilution, self.order, self.date)

    def load_data(self):
        """
        Creates the DataLoader objects for training
        """
        # create the custom datasets
        self.plaut_ds = Plaut_Dataset(self.plaut_filepath)
        self.anchor_ds = Plaut_Dataset(self.anchor_filepath)
        self.probe_ds = Plaut_Dataset(self.probe_filepath)
        
        # get the types to track from the dataset if not given
        if self.plaut_types == ['']:
            self.plaut_types = list(self.plaut_ds.get_types())
        else:
            assert set(self.plaut_types).issubset(self.plaut_ds.get_types()), "ERROR: Word types must exist in dataset."
        if self.anchor_types == ['']:
            self.anchor_types = list(self.anchor_ds.get_types())
        else:
            assert set(self.anchor_types).issubset(self.anchor_ds.get_types()), "ERROR: Word types must exist in dataset."
        if self.probe_types == ['']:
            self.probe_types = list(self.probe_ds.get_types())
        else:
            assert set(self.probe_types).issubset(self.probe_ds.get_types()), "ERROR: Word types must exist in dataset."

        # choose the specified anchor sets and set frequency appropriately
        self.anchor_ds.restrict_set(self.anchor_sets)
        self.anchor_ds.set_frequency(self.anchor_base_freq/len(self.anchor_sets))

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
        training_loss = Results(results_dir=self.rootdir+"/Training Loss", sim_label=self.label, title="Training Loss",
                                xlabel="Epoch", ylabel="Loss", anchor=self.anchor_epoch)
        plaut_accuracy = Results(results_dir=self.rootdir+"/Training Accuracy", sim_label=self.label, title="Training Accuracy",
                                 xlabel="Epoch", ylabel="Accuracy", categories=self.plaut_types, anchor=self.anchor_epoch)
        anchor_accuracy = Results(results_dir=self.rootdir+"/Anchor Accuracy", sim_label=self.label, title="Anchor Accuracy",
                                  xlabel="Epoch", ylabel="Accuracy", categories=self.anchor_types, anchor=self.anchor_epoch)
        probe_accuracy = Results(results_dir=self.rootdir+"/Probe Accuracy", sim_label=self.label, title="Probe Accuracy",
                                 xlabel="Epoch", ylabel="Accuracy", categories=self.probe_types, anchor=self.anchor_epoch)
        output_data = Results(results_dir=self.rootdir, sim_label=self.label, title="Simulation Results", xlabel='epoch',
                              categories=['example_id', 'orth', 'phon', 'category', 'correct'])
        time_data = Results(results_dir=self.rootdir, sim_label=self.label, title="Running Time", xlabel="Epoch", ylabel="Time (s)", anchor=self.anchor_epoch)
        
        """ TRAINING LOOP """
        for epoch in range(1, self.total_epochs+1):
            epoch_time = time.time()
            epoch_loss = 0

            # change optimizer if needed
            if epoch in self.optim_config['start_epoch']:
                optimizer = self.set_optimizer(self.current_optim)
                self.current_optim += 1

            # plaut dataset
            correct, total = np.zeros(len(self.plaut_types)), np.zeros(len(self.plaut_types))
            for i, data in enumerate(self.plaut_loader):
                loss, temp_correct, temp_total, compare = self.predict(data, categories=self.plaut_types) # find loss and accuracy

                epoch_loss += loss # accumulate loss
                correct += temp_correct # accumulate correct
                total += temp_total # accumulate total
            
            # save plaut accuracy results
            plaut_accuracy.append_row(epoch, (correct/total).tolist())
            
            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1, 1+self.plaut_samples)),
                'orth': data['orth'],
                'phon': data['phon'], 
                'category': data['type'],
                'correct': compare})
                #'anchors_added': 1 if epoch > self.anchor_epoch else 0})
            
            #output_data.append([epoch] * self.plaut_samples, [list(range(1, self.plaut_samples+1)),
                               #data['orth'], data['phon'], data['type'], compare*1])#, 0, self.dilution, self.order,
                               #self.random_seed, self.label, 0 if epoch < self.anchor_epoch else 1])
            
            # anchor dataset
            correct, total = np.zeros(len(self.anchor_types)), np.zeros(len(self.anchor_types))
            for i, data in enumerate(self.anchor_loader):
                loss, temp_correct, temp_total, compare = self.predict(data, categories=self.anchor_types) # find loss and accuracy

                # accumulate loss when anchors are added into training set
                if epoch > self.anchor_epoch:
                    epoch_loss += loss
                
                correct += temp_correct # accumulate correct
                total += temp_total # accumulate total
            
            # save anchor accuracy results
            anchor_accuracy.append_row(epoch, (correct/total).tolist())

            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1+self.plaut_samples, 1+self.plaut_samples+self.anchor_samples)),
                'orth': data['orth'],
                'phon': data['phon'], 
                'category': data['type'],
                'correct': compare})
                #'anchors_added': 1 if epoch > self.anchor_epoch else 0})
            
            # probe dataset
            correct, total = np.zeros(len(self.probe_types)), np.zeros(len(self.probe_types))
            for i, data in enumerate(self.probe_loader):
                loss, temp_correct, temp_total, compare = self.predict(data, categories=self.probe_types) # find loss and accuracy

                correct += temp_correct # accumulate correct
                total += temp_total # accumulate total
            
            output_data.add_rows([epoch] * len(compare), {
                'example_id': list(range(1+self.plaut_samples+self.anchor_samples, 1+self.plaut_samples+self.anchor_samples+self.probe_samples)),
                'orth': data['orth'],
                'phon': data['phon'], 
                'category': data['type'],
                'correct': compare})
                #'anchors_added': 1 if epoch > self.anchor_epoch else 0})
            
            # save probe accuracy results
            probe_accuracy.append_row(epoch, (correct/total).tolist())
            
            # calculate gradients and update weights
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # save loss results
            training_loss.append_row(epoch, epoch_loss.item())
            
            # plot results
            if epoch % self.plot_freq == 0:
                training_loss.lineplot()
                plaut_accuracy.lineplot()
                anchor_accuracy.lineplot()
                probe_accuracy.lineplot()

            # print statistics
            if epoch % self.print_freq == 0:
                epoch_time = time.time() - epoch_time
                print("[EPOCH {}] \t loss: {:.4f} \t time: {:.4f}".format(epoch, epoch_loss.item(), epoch_time))
                time_data.append_row(epoch, epoch_time)
        
        # save output data
        total_samples = self.plaut_samples + self.anchor_samples + self.probe_samples
        output_data.add_columns({
            'error': [0] * len(output_data),
            'random_seed': [self.random_seed] * len(output_data),
            'dilution': [self.dilution] * len(output_data),
            'order': [self.order] * len(output_data),
            'anchors_added': [0] * (self.anchor_epoch-1) * total_samples + [1] * (self.total_epochs + 1 - self.anchor_epoch) * total_samples
        })
        
        for key in ['optimizer', 'learning_rate', 'momentum', 'weight_decay']:
            temp = []
            for i in range(len(self.optim_config['start_epoch'])):
                epoch_start = int(self.optim_config['start_epoch'][i]) # start of optimizer config
                # end of optimizer config is next item in start, or if no more items, then end is total epochs
                try:
                    epoch_end = int(self.optim_config['start_epoch'][i+1])
                except:
                    epoch_end = self.total_epochs + 1
                for k in range(total_samples): # once per every word
                    temp += [self.optim_config[key][i] for j in range(epoch_start, epoch_end)] # once per every epoch
            output_data.add_columns({key: temp})

        output_data.save_data(index_label='Epoch')
        time_data.lineplot()
                      
    def set_optimizer(self, i):
        """
        Changes the optimizer based on configuration settings

        Arguments:
            i {int} -- optimizer index

        Returns:
            torch.optim.x.x -- the specified optimizer
        """
        if self.optim_config['optimizer'][i] == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.optim_config['learning_rate'][i], weight_decay=self.optim_config['weight_decay'][i]) 
        else:
            return optim.SGD(self.model.parameters(), lr=self.optim_config['learning_rate'][i], weight_decay=self.optim_config['weight_decay'][i], momentum=self.optim_config['momentum'][i]) 

    def predict(self, data, categories=['All']):
        """
        Makes predictions given the current model, as well as calculate loss and accuracy
        (Accuracy given as two seperate lists of correct and total)

        Arguments:
            data {dict} -- dictionary of data given by the DataLoader

        Keyword Arguments:
            categories {list} -- list of word categories to calculate accuracy over (default: {['All']})

        Returns:
            float -- loss
            np.array -- number of correct words per category
            np.array -- total number of words per category
            list -- True/False representing correctness of each word
        """        

        # extract log frequencies, grapheme vectors, phoneme vectors
        log_freq = data['log_freq'].view(-1, 1)
        inputs = data['graphemes']
        targets = data['phonemes']

        # forward pass + calculate loss
        outputs = self.model(inputs)
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
            if cat == 'All': # find accuracy across all categories
                correct.append(sum(compare)) / len(data['type'])
                total.append(len(data['type']))
            else:
                temp_total, temp_correct = 0, 0
                for t, c in zip(data['type'], compare):
                    if t == cat: # if same category
                        temp_total += 1
                        if c == True: # if correct
                            temp_correct += 1
                total.append(temp_total)
                correct.append(temp_correct)
        
        return loss, np.array(correct), np.array(total), compare


"""
TESTING AREA
"""

sim = Simulator("config.cfg")
sim.train()