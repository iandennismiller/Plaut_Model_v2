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
            ax.plot(self.values[key], label=key)


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
        df = pd.DataFrame(data=self.values, index=self.index)
        df.to_csv("{}/warping-dilution-{}.csv".format(self.results_dir, self.sim_label), index_label=index_label)
      

'''
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
        self.x = []
        self.y = [[] for i in categories]
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sim_label = sim_label
        self.categories = categories
        self.results_dir = results_dir
        self.anchor = anchor
        
    
    def __len__(self):
        """
        Returns the number of data points stored
        """
        return len(self.x)
    
    def append(self, x, y):
        """
        Adds a datapoint

        Arguments:
            x {list} -- list of the x-values of datapoint
            y {list} -- list of the lists of y-values of the datapoint
        """
        if type(x) != list:
            x = [x]
        
        if type(y) != list:
            y = [[y]]

        assert len(y) == len(self.categories), "ERROR: The y data given does not match the number of categories."

        # add datapoints
        self.x += x
        for i in range(len(self.categories)):
            if type(y[i]) == list:
                assert len(y[i]) == len(x), "ERROR: The length of {} given does not match the length of x".format(self.categories[i])
                self.y[i]+= y[i]
            else:
                self.y[i] += ([y[i]] * len(x))
    
    def lineplot(self, save=False):
        """
        Creates a Line Plot of all the datapoints

        NOTE: the data must be numeric for this function to work properly

        Keyword Arguments:
            save {bool} -- set to True to save a copy of the figure (default: {False})
        """
        fig, ax = plt.subplots()
        
        # plot for each category
        for i in range(len(self.categories)):
            ax.plot(self.x, self.y[i], label=self.categories[i])
        
        # axis labels, gridlines, and title
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(b=True, which='both', axis='both')
        plt.title(self.title)

        # add legend if needed
        if self.categories != [""]:
            plt.legend(loc='best')
        
        # add line for anchors if needed
        if max(self.x) > self.anchor:
            plt.axvline(x=self.anchor, color='red', lw=0.5)
        
        # annotate with simulation label
        plt.text(0.01, 0.01, self.sim_label, fontsize=8, transform=ax.transAxes)

        # ensure everything fits, save, and close
        plt.tight_layout()
        plt.savefig("{}/{} {:03d}.png".format(self.results_dir, self.title, self.x[-1]), dpi=200)
        plt.close()
    
    def save_data(self):
        results = {'label': self.sim_label, self.xlabel: self.x}
        for i in range(len(self.categories)):
            results[self.categories[i]] = self.y[i]

        df = pd.DataFrame(data=results)

        df.to_csv("{}/{}.csv".format(self.results_dir, self.title), index=False)
'''