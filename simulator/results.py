"""
results.py

=== SUMMARY ===
Description     : Class to store results and plot
Date Created    : May 04, 2020
Last Updated    : September 28, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1
    - fixed axis scaling is removed
    - method for plotting red line for anchor epoch is replaced, as such it does not appear in the legend
    - simulation label is added to plots

=== UPDATE NOTES ===
 > September 28, 2020
    - update function docstrings
    - add error checking for function arguments
 > September 27, 2020
    - add functonality of saving final plots in simulation root folder
 > September 9, 2020
    - parameter renaming
 > July 26, 2020
    - re-add compression for csv files
 > July 19, 2020
    - add changes for pd.Series values in add_rows function
    - update docstring
    - add label mapping for plots
 > July 18, 2020
    - minor reformatting changes
 > July 12, 2020
    - minor update to parameters due to changed configuration loading
 > May 24, 2020
    - annotate csv file with title
    - add .shape attribute
 > May 08, 2020
    - Update line_plot function to plot correct index values when checkpoints are used
    - Remove save parameter from line_plot function -> *must* be saved
    - Add function for creating and saving bar plots
 > May 07, 2020
    - Using pandas DataFrames result in slower run times in later iterations,
      so this is replaced with a dictionary as main results storage method
    - add compression on csv file
 > May 06, 2020
    - rewrite Results class using pandas DataFrame
 > May 05, 2020
    - add saving of plot
    - add sim_label to plot
    - add red line to plots for anchor epoch
 > May 04, 2020
    - file created
    - basic functionality of saving data in lists and plotting completed
"""

from config.simulator_config import Config

import pandas as pd
import logging
from matplotlib import pyplot as plt
import seaborn as sns


class Results:
    """
    Class for storing, plotting, and saving simulation data
    """

    def __init__(self, results_dir, config, title="", labels=("", ""), columns=None):
        """
        Initializes the Results class

        Arguments:
            results_dir (str)       : path to directory for saving plots and data
            config (Config)         : simulation config
            title (str)             : title used for plotting and sub-folder creation
            labels ((str, str))     : x and y-axis labels
            columns (list)          : column labels for y data series

        Returns:
            None
        """

        if columns is None:
            columns = [""]
        self.title = title
        self.x_label, self.y_label = labels
        self.sim_label = config.General.label
        self.results_dir = results_dir
        self.anchor = config.Training.anchor_epoch
        self.index = []
        self.values = {}
        for key in columns:
            self.values[key] = []
        self.shape = (0, len(columns))
        self.logger = logging.getLogger('__main__.' + __name__)

        sns.set_style('darkgrid')

    def __len__(self):
        """
        Returns the number of data points stored

        Returns:
            (int): number of stored data points
        """

        return len(self.index)

    def append_row(self, index, values):
        """
        Adds a **single** row to stored data

        **Notes**:
         - Use add_rows for adding multiple rows to stored data
         - If values is defined as list, user must ensure that value order matches key order

        Arguments:
            index (int)             : index of row to be added
            values (list or dict)   : list of values to be added

        Returns:
            None

        Raises:
            KeyError    : given data does not have the same keys as stored data
            ValueError  : given data is not the same length as keys of stored data
        """

        if type(values) != list:
            values = [values]

        self.index.append(index)  # add row index

        if type(values) == list:
            if len(values) != len(self.values.keys()):
                raise ValueError("Given values (list) does not have the same length as keys of stored data")
            for key, value in zip(self.values.keys(), values):  # add the values based on the respective keys
                self.values[key].append(value)
        elif type(values) == dict:
            if values.keys() != self.values.keys():
                raise KeyError("Keys of given data does not match keys of stored data")
            for key, value in values.items():
                self.values[key].append(value)

        self.shape = (self.shape[0] + 1, self.shape[1])

    def add_rows(self, indices, values_dict):
        """
        Adds multiple rows to the dataframe

        Arguments:
            indices (list)                                      : list of index values for the rows to be added
            values_dict (dict[str, list|pd.Series|int|float])   : dictionary containing the values to be added

        Returns:
            None

        Raises:
            KeyError    : given data does not have the same keys as stored data
            TypeError   : type of values in values_dict is not one of the accepted formats
        """

        if values_dict.keys() != self.values.keys():
            raise KeyError("Keys of given data does not match keys of stored data")

        self.index += indices  # add the row indices

        for key in values_dict.keys():  # add the values based on the respective keys
            if type(values_dict[key]) == list:
                assert len(values_dict[key]) == len(indices), f"Length of {key} values not equal to length of indices"
                self.values[key] += values_dict[key]
            elif type(values_dict[key]) == pd.Series:
                assert len(values_dict[key]) == len(indices), f"Length of {key} values not equal to length of indices"
                self.values[key] += values_dict[key].tolist()
            elif type(values_dict[key]) in [int, float]:
                self.values[key] += [values_dict[key]] * len(indices)
            else:
                raise TypeError(f"Type of values must be int, float, list, or pd.Series")

        self.shape = (self.shape[0] + len(indices), self.shape[1])

    def append_column(self, key, values):
        """
        Adds a **single** column to stored data

        **Notes**:
         - Use add_columns for adding multiple columns to stored data
         - If values is defined as list, user must ensure that value order matches key order

        Arguments:
            key (str)       : column label for the data
            values (list)   : values to be stored

        Returns:
            None

        Raises:
            ValueError  : Length of given values does not match length of stored data
        """

        if len(values) != self.shape[0]:
            raise ValueError('Length of given values does not match length of stored data')

        self.values[key] = values

        self.shape = (self.shape[0], self.shape[1] + 1)

    def add_columns(self, values_dict):
        """
        Adds multiple columns to the dictionary

        Arguments:
            values_dict (dict[str, list]): keys and values to be stored
        """
        self.values.update(values_dict)

        self.shape = (self.shape[0], self.shape[1] + 1)

    def line_plot(self, mapping=None, final=False):
        """
        Creates a Line Plot of all the data points

        Arguments:
            mapping: {dict} -- dictionary for label mapping from symbol to string

        NOTE: the data must be numeric for this function to work as intended
        """

        fig, ax = plt.subplots()

        # plot for each column
        for key in self.values.keys():
            ax.plot(self.index, self.values[key], label=mapping[key] if mapping else key)

        # axis labels, grid lines, and title
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid(b=True, which='both', axis='both')
        plt.title(self.title)

        # add legend if needed
        if len(self.values) > 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(self.values.keys()))

        # add line for anchors if needed
        if max(self.index) > self.anchor:
            plt.axvline(x=self.anchor, color='red', lw=0.5)

        # annotate with simulation label
        plt.text(0.01, 0.01, self.sim_label, fontsize=8, transform=ax.transAxes)

        # ensure everything fits, save, and close
        plt.tight_layout()
        if final:
            parent_dir = '/'.join(self.results_dir.split('/')[:-1])
            plt.savefig(f"{parent_dir}/{self.title}.png", dpi=200)
        else:
            plt.savefig(f"{self.results_dir}/{self.title} {max(self.index):04d}.png", dpi=200)
        plt.close()

    def bar_plot(self, final=False):
        """
        Creates a Bar Plot for current epoch

        NOTE: the data must be numeric for this function to work as intended
        """
        # create figure
        plt.figure()

        # plot bars
        for key in self.values.keys():
            plt.bar(key, self.values[key][-1])

        # axis labels and title
        plt.xlabel('Category')
        plt.ylabel(self.y_label)
        plt.title(self.title)

        # ensure everything fits, save, and close
        plt.tight_layout()
        if final:
            parent_dir = '/'.join(self.results_dir.split('/')[:-1])
            plt.savefig(f"{parent_dir}/{self.title} Bar.png", dpi=200)
        else:
            plt.savefig(f"{self.results_dir}/{self.title} Bar {max(self.index):04d}.png", dpi=200)
        plt.close()

    def save_data(self, index_label="epoch", save_type='csv'):
        """
        Saves results as compressed csv file

        Keyword Arguments:
            index_label {str} -- label for index column (default: {"epoch"})
        """

        df = pd.DataFrame(data=self.values, index=self.index)  # create pandas dataframe
        if save_type == 'csv':
            df.to_csv(f"{self.results_dir}/warping-dilution-{self.sim_label}-{self.title}.csv.gz",
                      index_label=index_label)  # save as compressed csv
            self.logger.info(f'{self.title} saved successfully as compressed csv')
        elif save_type == 'pickle':

            df.to_pickle(f"{self.results_dir}/warping-dilution-{self.sim_label}-{self.title}.pkl")
            self.logger.info(f'{self.title} saved successfully as pickle')
