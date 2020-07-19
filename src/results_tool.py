"""
results_tool.py

=== SUMMARY ===
Description     : Class to store results and plot
Date Created    : May 04, 2020
Last Updated    : July 19, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1
    - fixed axis scaling is removed
    - method for plotting red line for anchor epoch is replaced, as such it does not appear in the legend
    - simulation label is added to plots

=== UPDATE NOTES ===
 > July 19, 2020
    - add changes for pd.Series values in add_rows function
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

from matplotlib import pyplot as plt
import pandas as pd


class Results:
    def __init__(self, results_dir, config, title="", labels=("", ""), categories=None):
        """
        Initializes a class for storing, plotting, and saving data

        Arguments:
            results_dir {str} -- path to directory for saving plots and data

        Keyword Arguments:
            sim_label {str} -- simulation label for annotating plots (default: {""})
            title {str} -- title for plotting (default: {""})
            labels {str} -- x-axis and y-axis label for plotting (default: {("", "")})
            categories {list} -- categories of y data (default: {None})
            anchor {int} -- epoch that anchors are added, for annotation on plot (default: {None})
        """

        if categories is None:
            categories = [""]
        self.title = title
        self.x_label, self.y_label = labels
        self.sim_label = config.General.label
        self.results_dir = results_dir
        self.anchor = config.Training.anchor_epoch
        self.index = []
        self.values = {}
        for key in categories:
            self.values[key] = []
        self.shape = (0, len(categories))

    def __len__(self):
        """
        Returns the number of data points stored
        """

        return len(self.index)

    def append_row(self, index, values):
        """
        Adds a *single* row to the dataframe, use add_rows for multiple rows

        Arguments:
            index {int} -- index of row to be added
            values {list} -- list of values to be added
        """

        if type(values) != list:
            values = [values]

        self.index.append(index)  # add row index

        for key, value in zip(self.values.keys(), values):  # add the values based on the respective keys
            self.values[key].append(value)

        self.shape = (self.shape[0] + 1, self.shape[1])

    def add_rows(self, indices, values_dict):
        """
        Adds multiple rows to the dataframe

        Arguments:
            indices {list} -- list of index values for the rows to be added
            values_dict {dict} -- dictionary containing the values to be added
        """

        assert set(values_dict.keys()) == set(self.values.keys())  # assert that keys match

        self.index += indices  # add the row indices

        for key in values_dict.keys():  # add the values based on the respective keys
            if type(values_dict[key]) == list:
                assert len(values_dict[key]) == len(indices), \
                    f"Length of {key} values not equal to length of indices"
                self.values[key] += values_dict[key]
            elif type(values_dict[key]) == pd.Series:
                assert len(values_dict[key]) == len(indices), \
                    f"Length of {key} values not equal to length of indices"
                self.values[key] += values_dict[key].tolist()
            elif type(values_dict[key]) in [int, float]:
                self.values[key] += [values_dict[key]] * len(indices)
            else:
                raise TypeError(f"Type of values must be int, float, list, or pd.Series")

        self.shape = (self.shape[0] + len(indices), self.shape[1])

    def append_column(self, label, values):
        """
        Adds a *single* column to the dictionary, use add_columns for multiple columns

        Arguments:
            label {str} -- column label for the data
            values {list} -- list of values to be added
        """

        self.values[label] = values

        self.shape = (self.shape[0], self.shape[1] + 1)

    def add_columns(self, values_dict):
        """
        Adds multiple columns to the dictionary

        Arguments:
            values_dict {[type]} -- dictionary consisting of keys and values representing the columns to be added
        """
        self.values.update(values_dict)

        self.shape = (self.shape[0], self.shape[1] + 1)

    def line_plot(self):
        """
        Creates a Line Plot of all the data points

        NOTE: the data must be numeric for this function to work as intended
        """

        fig, ax = plt.subplots()

        # plot for each column
        for key in self.values.keys():
            ax.plot(self.index, self.values[key], label=key)

        # axis labels, grid lines, and title
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid(b=True, which='both', axis='both')
        plt.title(self.title)

        # add legend if needed
        if len(self.values) > 1:
            plt.legend(loc='best')

        # add line for anchors if needed
        if max(self.index) > self.anchor:
            plt.axvline(x=self.anchor, color='red', lw=0.5)

        # annotate with simulation label
        plt.text(0.01, 0.01, self.sim_label, fontsize=8, transform=ax.transAxes)

        # ensure everything fits, save, and close
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/{self.title} {max(self.index):03d}.png", dpi=200)
        plt.close()

    def bar_plot(self):
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
        plt.savefig(f"{self.results_dir}/{self.title} Bar {max(self.index):03d}.png", dpi=200)
        plt.close()

    def save_data(self, index_label="epoch"):
        """
        Saves results as compressed csv file

        Keyword Arguments:
            index_label {str} -- label for index column (default: {"epoch"})
        """

        df = pd.DataFrame(data=self.values, index=self.index)  # create pandas dataframe
        df.to_csv(f"{self.results_dir}/warping-dilution-{self.sim_label}-{self.title}.csv",
                  index_label=index_label)  # save as compressed csv
