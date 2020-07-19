"""
dataset.py

=== SUMMARY === 
Description     : Defines class for custom dataset for Plaut Model and functions to generate grapheme/phoneme vectors
Date Created    : May 03, 2020
Last Updated    : July 18, 2020

=== DETAILED DESCRIPTION ===
 > Functionality
   1. Takes a csv file with orthography and phonology of words and converts to vector format
 > Changes from v1
   - in plaut_dataset, convert freq/log_freq to float at initialization rather than at sampling
   - in plaut_dataset.__getitem__, return {type, orth, phon} as lists instead of pandas Series

=== UPDATE NOTES ===
 > July 18, 2020
    - minor typo fixes and formatting changes
 > May 03, 2020
    - File created, initial code written
"""

import pandas as pd
import numpy as np
import torch as torch
from torch.utils.data import Dataset

# MAPPINGS FOR GRAPHEMES (30 onset, 27 vowel, 48 codas for total of 105)
grapheme_onset = ['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G', 'F', 'V', 'J', 'Z',
                  'L', 'M', 'N', 'R', 'W', 'H', 'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH']
grapheme_vowel = ['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 'EA', 'EE', 'EI',
                  'EU', 'EW', 'EY', 'IE', 'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY']
grapheme_codas = ['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 'F', 'V', 'J', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB',
                  'CH', 'CK', 'DD', 'DG',
                  'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL', 'SS', 'TCH', 'TH',
                  'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED']

# MAPPINGS FOR PHONEMES (23 onset, 14 vowel, 24 codas for total of 61)
phoneme_onset = ['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D',
                 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'l', 'r', 'w', 'y']
phoneme_vowel = ['a', 'e', 'i', 'o', 'u', '@',
                 '^', 'A', 'E', 'I', 'O', 'U', 'W', 'Y']
phoneme_codas = ['r', 'l', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks',
                 'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z', 'T', 'D', 'C', 'j']


class PlautDataset(Dataset):
    def __init__(self, filepath):
        """
        Initializes a dataset, calculates log_freq column if required,

        Detailed Description:
         - loads a dataset from file into pandas DataFrame
         - calculates log frequency (i.e. ln(freq+2)) if column does not already exist
         - converts orthography and phonology to vector form

        Arguments:
            filepath {str} -- filepath to the file containing orthography, phonology, type, and frequency
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
            int -- total number of samples
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Obtains samples at the specified index/indices

        Arguments:
            index {list} -- list specifying indices of samples desired (can also be tensor)

        Returns:
            dict -- dictionary of word type, orthography, phonology, log frequency, grapheme vector and phoneme vector
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
            new_freq {int/float} -- new desired frequency
        """
        self.df["freq"] = float(new_freq)
        self.df["log_freq"] = np.log(new_freq + 2)

    def scale_frequency(self, factor):
        """
        Scales frequency of all words by a fixed factor
        Note: new_freq = factor * old_freq

        Arguments:
            factor {int/float} -- factor to scale frequencies by
        """
        self.df["freq"] = self.df["freq"] * factor
        self.df["log_freq"] = np.log(self.df["freq"])

    def restrict_set(self, sets):
        assert 'set' in self.df.columns, "ERROR: Dataset cannot be restricted by sets"
        self.df = self.df.loc[self.df['set'].isin(sets)]
        self.df.reset_index(drop=True, inplace=True)

    def get_types(self):
        return set(self.df['type'])


def get_graphemes(word):
    word = str(word).upper()  # convert all text to capitals first
    if word == "NAN":  # the word null automatically gets imported as "NaN" in dataframe, so fix that
        word = "NULL"

    # initialize vectors to zero
    onset = [0] * len(grapheme_onset)
    vowel = [0] * len(grapheme_vowel)
    codas = [0] * len(grapheme_codas)

    # for onset: essentially "turn on" corresponding slots for onsets until a vowel is reached
    i = 0
    for i in range(len(word)):
        if word[i] in grapheme_vowel:  # vowel found, move on
            if not (i == 0 and word[i] == 'Y'):
                break
        if word[i] in grapheme_onset:  # single-letter grapheme found
            onset[grapheme_onset.index(word[i])] = 1
        if word[i:i + 2] in grapheme_onset:  # double-letter grapheme found
            onset[grapheme_onset.index(word[i:i + 2])] = 1

    # for vowels
    vowel[grapheme_vowel.index(word[i])] = 1
    if i + 1 < len(word):  # check for double-vowel
        if word[i + 1] in grapheme_vowel:
            vowel[grapheme_vowel.index(word[i + 1])] = 1
        if word[i:i + 2] in grapheme_vowel:
            vowel[grapheme_vowel.index(word[i:i + 2])] = 1
        # if double-letter vowel found, increment i one more time
        if word[i + 1] in grapheme_vowel or word[i:i + 2] in grapheme_vowel:
            i += 1

    # for codas
    for j in range(i + 1, len(word)):
        if word[j] in grapheme_codas:  # check for single-letter coda
            codas[grapheme_codas.index(word[j])] = 1
        if word[j:j + 2] in grapheme_codas:  # check for double-letter coda
            codas[grapheme_codas.index(word[j:j + 2])] = 1
        if word[j:j + 3] in grapheme_codas:  # check for triple-letter coda
            codas[grapheme_codas.index(word[j:j + 3])] = 1

    # combine and return
    return onset + vowel + codas


# similar idea to graphemes; refer to above for comments
def get_phonemes(phon):
    phon = phon[1:-1]
    onset = [0] * len(phoneme_onset)
    vowel = [0] * len(phoneme_vowel)
    codas = [0] * len(phoneme_codas)

    i, j, k = 0, 0, 0
    for i in range(len(phon)):
        if phon[i] in phoneme_vowel:
            break
        if phon[i] in phoneme_onset:
            onset[phoneme_onset.index(phon[i])] = 1

    for j in range(i, len(phon)):
        if phon[j] in phoneme_codas:
            break
        if phon[j] in phoneme_vowel:
            vowel[phoneme_vowel.index(phon[j])] = 1

    for k in range(j, len(phon)):
        if phon[k] in phoneme_codas:
            codas[phoneme_codas.index(phon[k])] = 1
        if phon[k:k + 2] in phoneme_codas:
            codas[phoneme_codas.index(phon[k:k + 2])] = 1

    return onset + vowel + codas
