"""
density_plots.py

=== SUMMARY ===
Description     : Density plots for hidden layer activations
Date Created    : June 27, 2020
Last Updated    : July 27, 2020

=== UPDATE NOTES ===
 > July 27, 2020
    - update functions, and put into class
 > June 27, 2020
    - file created
"""

import pandas as pd
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from common.constants import WordTypes, VectorMapping
from common.helpers import *
from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


class DensityPlots:
    def __init__(self, folder, plaut=False, anchor=False, probe=False):
        self.results_dir = folder
        self.label = folder.split('/')[-1]
        self.categories = []
        if plaut:
            self.categories += WordTypes.plaut_types
        if anchor:
            self.categories += WordTypes.anchor_types
        if probe:
            self.categories += WordTypes.probe_types

        self.output_dir = create_analysis_folder(self.results_dir, 'density_plots')
        self.logger = logging.getLogger('__main__.' + __name__)
        self.logger.info(f'Categories: {self.categories}')

    def create_hl_plots(self):
        # CHECK WHETHER DATA FILE EXISTS
        filepath = f'{self.results_dir}/warping-dilution-{self.label}-Hidden Layer.csv.gz'
        if not os.path.isfile(filepath):
            self.logger.error('No data found for hidden layer activations in given folder')
            return None

        # LOAD DATA
        self.logger.info('Loading hidden layer activation data')
        df = pd.read_csv(filepath, converters={'activation': eval})
        df = df[df['category'].isin(self.categories)]

        # CREATE DENSITY PLOTS
        ymin, ymax = None, None
        t = tqdm(sorted(df['epoch'].unique(), reverse=True))
        for epoch in t:
            t.set_description(f"[Hidden Layer Activation - Epoch {epoch:3d}]")
            epoch_df = df[df['epoch'] == epoch]
            fig = plt.figure()
            for cat in self.categories:
                if cat in WordTypes.anchor_types:
                    label = WordTypes.anchor_mapping[cat]
                elif cat in WordTypes.probe_types:
                    label = WordTypes.probe_mapping[cat]
                else:
                    label = cat

                activation_data = epoch_df[epoch_df['category'] == cat]['activation'].apply(pd.Series)
                data = activation_data.to_numpy().reshape(-1)
                sns.distplot(data, hist=None, kde_kws={'bw': 0.025, 'gridsize': 150}, label=label)

            if ymin is None:
                xmin, xmax, ymin, ymax = plt.axis()
            else:
                plt.ylim(ymin, ymax)

            plt.title(f'Hidden Layer Activations - Epoch {epoch}')
            plt.legend()
            plt.savefig(f'{self.output_dir}/hidden_layer_activations_{epoch}', dpi=300)
            plt.close()

        self.logger.info("Completed all required density plots for hidden layer activations")

    def create_ol_plots(self):
        # CHECK WHETHER DATA FILE EXISTS
        filepath = f'{self.results_dir}/warping-dilution-{self.label}-Output Layer.csv.gz'
        if not os.path.isfile(filepath):
            logging.error('No data found for output layer activations in given folder')
            return None

        # LOAD DATA
        self.logger.info('Loading output layer activation data')
        df = pd.read_csv(filepath, converters={'activation': eval})
        df = df[df['category'].isin(self.categories)]

        num_onset = len(VectorMapping.phoneme_onset)
        num_vowel = len(VectorMapping.phoneme_vowel)

        # CREATE DENSITY PLOTS
        t = tqdm(df['epoch'].unique())
        for epoch in t:
            t.set_description(f"[Output Layer Activation - Epoch {epoch:3d}]")
            epoch_df = df[df['epoch'] == epoch]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharex='all', sharey='all')

            labels = []
            for cat in self.categories:
                if cat in WordTypes.anchor_types:
                    label = WordTypes.anchor_mapping[cat]
                elif cat in WordTypes.probe_types:
                    label = WordTypes.probe_mapping[cat]
                else:
                    label = cat
                labels.append(label)

                activation_data = epoch_df[epoch_df['category'] == cat]['activation'].apply(pd.Series)
                data = activation_data.to_numpy()
                ax1 = sns.distplot(data[:, :num_onset].reshape(-1), hist=None,
                                   kde_kws={'bw': 0.025, 'gridsize': 150}, ax=ax1)
                ax2 = sns.distplot(data[:, num_onset:num_onset + num_vowel].reshape(-1), hist=None,
                                   kde_kws={'bw': 0.025, 'gridsize': 150}, ax=ax2)
                ax3 = sns.distplot(data[:, num_onset + num_vowel:].reshape(-1), hist=None,
                                   kde_kws={'bw': 0.025, 'gridsize': 150}, label=label, ax=ax3)

            ax1.set_title("Onsets")
            ax2.set_title("Vowels")
            ax3.set_title("Codas")
            plt.suptitle(f'Output Layer Activations - Epoch {epoch}')
            plt.subplots_adjust(left=0.05, right=0.8)
            ax3.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
            plt.xlim(-0.1, 1.1)
            plt.ylim(0, 1.5)
            plt.savefig(f'{self.output_dir}/output_layer_activations_{epoch}', dpi=300)
            plt.close()

        self.logger.info("Completed all required density plots for output layer activations")
