"""
results_tool.py

=== SUMMARY ===
Description     : Class to store results and plot
Date Created    : May 04, 2020
Last Updated    : May 04, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1
    - fixed axis scaling is removed
    - method for plotting red line for anchor epoch is replaced, as such it does not appear in the legend
    - simulation label is added to plots

=== UPDATE NOTES ===
 > May 08, 2020
    - Update lineplot function to plot correct index values when checkpoints are used
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


class Results():
    def __init__(self, results_dir, sim_label="", title="", xlabel="", ylabel="", categories=[""], anchor=None):
        """
        Initializes a class for storing, plotting, and saving data

        Arguments:
            results_dir {str} -- path to directory for saving plots and data

        Keyword Arguments:
            sim_label {str} -- simulation label for annotating plots (default: {""})
            title {str} -- title for plotting (default: {""})
            xlabel {str} -- x-axis label for plotting (default: {""})
            ylabel {str} -- y-axis label for plotting (default: {""})
            categories {list} -- categories of y data (default: {[""]})
            anchor {int} -- epoch that anchors are added, for annotation on plot (default: {None})
        """
        
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sim_label = sim_label
        self.results_dir = results_dir
        self.anchor = anchor
        self.index = []
        self.values = {}
        for key in categories:
            self.values[key] = []

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

        self.index.append(index) # add row index
        
        for key, value in zip(self.values.keys(), values): # add the values based on the respective keys
            self.values[key].append(value)
    
    def add_rows(self, indices, values_dict):
        """
        Adds multiple rows to the dataframe

        Arguments:
            indices {list} -- list of index values for the rows to be added
            values_dict {dict} -- dictionary containing the values to be added
        """

        assert set(values_dict.keys()) == set(self.values.keys())
        
        self.index += indices # add the row indices
        
        for key in values_dict.keys(): # add the values based on the respective keys
            if type(values_dict[key]) == list:
                assert len(values_dict[key]) == len(indices), f"ERROR: Length of values for {key} is not equal to length of indices"
                self.values[key] += values_dict[key]
            else:
                self.values[key] += [values_dict[key]] * len(indices)
    
    def append_column(self, label, values):
        """
        Adds a *single* column to the dictionary, use add_columns for multiple columns

        Arguments:
            label {str} -- column label for the data
            values {list} -- list of values to be added
        """

        self.values[label] = values
    
    def add_columns(self, values_dict):
        """
        Adds multiple columns to the dictionary

        Arguments:
            values_dict {[type]} -- dictionary consisting of keys and values representing the columns to be added
        """
        self.values.update(values_dict)

    def lineplot(self, save=False):
        """
        Creates a Line Plot of all the datapoints

        NOTE: the data must be numeric for this function to work properly

        Keyword Arguments:
            save {bool} -- set to True to save a copy of the figure (default: {False})
        """
        
        fig, ax = plt.subplots()
        
        # plot for each column
        for key in self.values.keys():
            ax.plot(self.index, self.values[key], label=key)


        # axis labels, gridlines, and title
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
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
        plt.savefig("{}/{} {:03d}.png".format(self.results_dir, self.title, max(self.index)), dpi=200)
        plt.close()
    
    def save_data(self, index_label="epoch"):
        """
        Saves results as compressed csv file

        Keyword Arguments:
            index_label {str} -- label for index column (default: {"epoch"})
        """
        df = pd.DataFrame(data=self.values, index=self.index) # create pandas dataframe
        df.to_csv("{}/warping-dilution-{}.csv.gz".format(self.results_dir, self.sim_label), index_label=index_label) #save as compressed csv