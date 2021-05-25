"""
dataset.py

=== SUMMARY === 
Description     : Defines class for custom dataset for Plaut Model and functions to generate grapheme/phoneme vectors
Date Created    : May 03, 2020
Last Updated    : September 28, 2020

=== DETAILED DESCRIPTION ===
 > Functionality
   1. Takes a csv file with orthography and phonology of words and converts to vector format
 > Changes from v1
   - in plaut_dataset, convert freq/log_freq to float at initialization rather than at sampling
   - in plaut_dataset.__getitem__, return {type, orth, phon} as lists instead of pandas Series

=== UPDATE NOTES ===
 > September 28, 2020
    - update function docstrings
 > August 4, 2020
    - move get_grapheme and get_phoneme functions to common/helpers.py
 > July 19, 2020
    - move grapheme and phoneme mapping to constants class
 > July 18, 2020
    - minor typo fixes and formatting changes
 > May 03, 2020
    - File created, initial code written
"""

from common.helpers import *

import numpy as np
import pandas as pd
import torch as torch
from torch.utils.data import Dataset


class PlautDataset(Dataset):
    def __init__(self, filepath):
        """
        Initializes a dataset, calculates log_freq column if required,

        Detailed Description:
         - loads a dataset from file into pandas DataFrame
         - calculates log frequency (i.e. ln(freq+2)) if column does not already exist
         - converts orthography and phonology to vector form

        Arguments:
            filepath (str): filepath to the file containing orthography, phonology, type, and frequency
        """

        # load dataset
        self.df = pd.read_csv(filepath, na_filter=False, dtype={'freq': float, 'log_freq': float})

        # ensure dataset must have the required columns
        assert {'orth', 'phon', 'type'}.issubset(set(self.df.columns)), \
            "ERROR: Dataset file does not contain the minimum required columns"

        if 'freq' not in self.df.columns:  # if frequency not given, use placeholder of 0
            self.set_frequency(0)
        elif 'log_freq' not in self.df.columns:  # calculate log frequency if column does not already exist
            self.df['log_freq'] = np.log(self.df['freq'] + 2)

        self.df["graphemes"] = self.df["orth"].apply(lambda x: get_graphemes(x))
        self.df["phonemes"] = self.df["phon"].apply(lambda x: get_phonemes(x))

    def __len__(self):
        """
        Return the number of samples in dataframe

        Returns:
            (int): total number of samples
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Obtains samples at the specified index/indices

        Arguments:
            index (list or torch.Tensor): indices of samples desired (can also be tensor)

        Returns:
            (dict): dictionary of word type, orthography, phonology, log frequency, grapheme vector and phoneme vector
        """
        if torch.is_tensor(index):  # convert to list if tensor given
            index = index.tolist()

        return {"type": self.df.loc[index, "type"],  # type of word
                "orth": self.df.loc[index, "orth"],  # orthography (e.g. ace)
                "phon": self.df.loc[index, "phon"],  # phonology (e.g. /As/)
                "log_freq": torch.tensor(self.df.loc[index, "log_freq"], dtype=torch.float),
                # the frequency AFTER log transform
                "graphemes": torch.tensor(self.df.loc[index, 'graphemes'], dtype=torch.float),
                # vector of graphemes representing orthography
                "phonemes": torch.tensor(self.df.loc[index, 'phonemes'], dtype=torch.float)
                # vector of phonemes representing phonology
                }

    def set_frequency(self, new_freq):
        """
        Sets a new, fixed frequency for all words

        Arguments:
            new_freq (int or float): new desired frequency

        Returns:
            None
        """
        self.df["freq"] = float(new_freq)
        self.df["log_freq"] = np.log(new_freq + 2)

    def scale_frequency(self, factor):
        """
        Scales frequency of all words by a fixed factor

        Note: new_freq = factor * old_freq

        Arguments:
            factor (int or float): factor to scale frequencies by

        Returns:
            None
        """
        self.df["freq"] = self.df["freq"] * factor
        self.df["log_freq"] = np.log(self.df["freq"])

    def restrict_set(self, sets):
        """
        Restrict anchor sets for dilution factor

        Arguments:
            sets (list): set indices to be kept

        Returns:
            None
        """
        assert 'set' in self.df.columns, "ERROR: Dataset cannot be restricted by sets"
        self.df = self.df.loc[self.df['set'].isin(sets)]
        self.df.reset_index(drop=True, inplace=True)

    def get_types(self):
        """
        Find the word types (categories) in the dataset

        Returns:
            (set): set of word types (categories) in the dataset
        """
        return set(self.df['type'])

