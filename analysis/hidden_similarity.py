"""
hidden_similarity.py

=== SUMMARY ===
Description     : Analyze how the hideen representations are organized on the basis of orthographic similarity
                  (see Plaut paper Figure 18)
Date Created    : August 30, 2020
Last Updated    : August 30, 2020

=== UPDATE NOTES ===
 > August 30, 2020
    - file created
"""

from simulator.model import PlautNet
from common.helpers import *
from common.constants import PlotTypes
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import logging

plot_colours = {
    'regular': '#ff7f0e',
    'exception': '#2ca02c',
    'nonword': '#1f77b4'
}
plot_linestyle = {
    'orth_phon_corr': '-',
    'orth_hidden_corr': '--',
    'hidden_phon_corr': ':'
}
plot_labels = {
    'orth_phon_corr': 'Orth-Phon',
    'orth_hidden_corr': 'Orth-Hidden',
    'hidden_phon_corr': 'Hidden-Phon'
}


class HiddenSimilarity:
    def __init__(self, checkpoint_label, checkpoint2_label=None):
        self.logger = logging.getLogger('__main__.' + __name__)
        self.model = PlautNet()

        self.checkpoint, self.epoch = None, None
        self.checkpoint2, self.epoch2 = None, None
        self.result_dir, self.result_dir = None, None
        self.label = checkpoint_label.split('-')[0]+'_'+checkpoint_label.split('-')[-1]

        self.result_dir = f'results/{checkpoint_label}'
        self.checkpoint, self.epoch = self.load_checkpoints(checkpoint_label)
        if checkpoint2_label:
            self.result_dir2 = f'results/{checkpoint2_label}'
            self.checkpoint2, self.epoch2 = self.load_checkpoints(checkpoint2_label)
            if self.epoch != self.epoch2:
                self.logger.error("Checkpoint epochs must be identical between the two checkpoint sets.")
                quit()

    def load_checkpoints(self, label):
        all_checkpoints = os.listdir('checkpoints')
        checkpoint = [x for x in all_checkpoints if label in x]
        if not checkpoint:
            self.logger.error(f"No checkpoint files exist for given label {label}")
            quit()
        checkpoint = sorted(checkpoint, key=lambda x: int(x.rstrip('.tar').split('_')[-1]))
        epoch = [int(x.rstrip('.tar').split('_')[-1]) for x in checkpoint]
        checkpoint = [torch.load(f"checkpoints/{x}") for x in checkpoint]
        return checkpoint, epoch

    @staticmethod
    def calculate_correlation_vectors(df):
        dfs = []

        for epoch in df['epoch'].unique():
            temp = df[df['epoch'] == epoch].reset_index(drop=True)
            corr_df = temp['graphemes'].apply(pd.Series).T.corr()
            temp['graphemes_corr_vector'] = corr_df.values.tolist()
            temp['graphemes_corr_vector'] = temp.apply(
                lambda row: row['graphemes_corr_vector'][:row.name] + row['graphemes_corr_vector'][row.name+1:], axis=1)

            corr_df = temp['hidden'].apply(pd.Series).T.corr()
            temp['hidden_corr_vector'] = corr_df.values.tolist()
            temp['hidden_corr_vector'] = temp.apply(
                lambda row: row['hidden_corr_vector'][:row.name] + row['hidden_corr_vector'][row.name + 1:], axis=1)

            corr_df = temp['phonemes'].apply(pd.Series).T.corr()
            temp['phonemes_corr_vector'] = corr_df.values.tolist()
            temp['phonemes_corr_vector'] = temp.apply(
                lambda row: row['phonemes_corr_vector'][:row.name] + row['phonemes_corr_vector'][row.name + 1:], axis=1)
            dfs.append(temp)

        return pd.concat(dfs)

    @staticmethod
    def calculate_cross_layer_correlation(df):
        df['orth_hidden_corr'] = df.apply(lambda row: np.corrcoef(row['graphemes_corr_vector'],
                                                                  row['hidden_corr_vector'])[0, 1], axis=1)
        df['hidden_phon_corr'] = df.apply(lambda row: np.corrcoef(row['hidden_corr_vector'],
                                                                  row['phonemes_corr_vector'])[0, 1], axis=1)
        df['orth_phon_corr'] = df.apply(lambda row: np.corrcoef(row['graphemes_corr_vector'],
                                                                row['phonemes_corr_vector'])[0, 1], axis=1)
        return df

    def find_graphemes_hidden_phonemes(self, df, checkpoint, epoch):
        df['graphemes'] = df['orth'].apply(lambda x: get_graphemes(x))

        dfs = []
        for c, e in zip(checkpoint, epoch):
            self.model.load_state_dict(c)
            self.model.eval()
            hidden, phon = self.model(torch.tensor(df['graphemes'], dtype=torch.float))
            dfs.append(pd.DataFrame(data={'orth': df['orth'],
                                          'epoch': [e] * len(df),
                                          'graphemes': df['graphemes'],
                                          'hidden': hidden.tolist(),
                                          'phonemes': phon.tolist()}))

        return pd.concat(dfs).reset_index(drop=True)

    def create_plots(self, dataset_filename):
        test_dataset = pd.read_csv(f"dataset/{dataset_filename}")
        regulars2, exceptions2, nonwords2 = None, None, None

        # extract orthology of words
        regulars = test_dataset['Regular Inconsistent'].to_frame().rename(columns={'Regular Inconsistent': 'orth'})
        exceptions = test_dataset['Exception'].to_frame().rename(columns={'Exception': 'orth'})
        nonwords = test_dataset['Nonword'].to_frame().rename(columns={'Nonword': 'orth'})

        # find graphemes, hidden, phonemes
        regulars = self.find_graphemes_hidden_phonemes(regulars, self.checkpoint, self.epoch)
        exceptions = self.find_graphemes_hidden_phonemes(exceptions, self.checkpoint, self.epoch)
        nonwords = self.find_graphemes_hidden_phonemes(nonwords, self.checkpoint, self.epoch)

        # calculate correlation for same layer between words
        regulars = self.calculate_correlation_vectors(regulars)
        exceptions = self.calculate_correlation_vectors(exceptions)
        nonwords = self.calculate_correlation_vectors(nonwords)

        # calculate correlation between layers for same word
        regulars = self.calculate_cross_layer_correlation(regulars)
        exceptions = self.calculate_cross_layer_correlation(exceptions)
        nonwords = self.calculate_cross_layer_correlation(nonwords)

        if self.checkpoint2:
            regulars2 = test_dataset['Regular Inconsistent'].to_frame().rename(columns={'Regular Inconsistent': 'orth'})
            exceptions2 = test_dataset['Exception'].to_frame().rename(columns={'Exception': 'orth'})
            nonwords2 = test_dataset['Nonword'].to_frame().rename(columns={'Nonword': 'orth'})

            regulars2 = self.find_graphemes_hidden_phonemes(regulars2, self.checkpoint2, self.epoch2)
            exceptions2 = self.find_graphemes_hidden_phonemes(exceptions2, self.checkpoint2, self.epoch2)
            nonwords2 = self.find_graphemes_hidden_phonemes(nonwords2, self.checkpoint2, self.epoch2)

            regulars2 = self.calculate_correlation_vectors(regulars2)
            exceptions2 = self.calculate_correlation_vectors(exceptions2)
            nonwords2 = self.calculate_correlation_vectors(nonwords2)

            regulars2 = self.calculate_cross_layer_correlation(regulars2)
            exceptions2 = self.calculate_cross_layer_correlation(exceptions2)
            nonwords2 = self.calculate_cross_layer_correlation(nonwords2)


        # average and plot
        cols = ['epoch', 'orth_phon_corr', 'orth_hidden_corr', 'hidden_phon_corr']
        regulars = regulars[cols].groupby(by='epoch').mean()
        exceptions = exceptions[cols].groupby(by='epoch').mean()
        nonwords = nonwords[cols].groupby(by='epoch').mean()

        if self.checkpoint2:
            regulars2 = regulars2[cols].groupby(by='epoch').mean()
            exceptions2 = exceptions2[cols].groupby(by='epoch').mean()
            nonwords2 = nonwords2[cols].groupby(by='epoch').mean()
            regulars = pd.concat([regulars, regulars2]).groupby(by='epoch').mean()
            exceptions = pd.concat([exceptions, exceptions2]).groupby(by='epoch').mean()
            nonwords = pd.concat([nonwords, nonwords2]).groupby(by='epoch').mean()

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(9, 6))
        for i in ['orth_phon_corr', 'orth_hidden_corr', 'hidden_phon_corr']:
            regulars[i].plot.line(ax=ax, color=plot_colours['regular'], marker='.',
                                  linestyle=plot_linestyle[i], label=f'{plot_labels[i]} | Regular')
            exceptions[i].plot.line(ax=ax, color=plot_colours['exception'], marker='.',
                                    linestyle=plot_linestyle[i], label=f'{plot_labels[i]} | Exception')
            nonwords[i].plot.line(ax=ax, color=plot_colours['nonword'], marker='.',
                                  linestyle=plot_linestyle[i], label=f'{plot_labels[i]} | Nonword')

        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.ylim(0.5, 1)
        plt.suptitle('Similarity Correlations (Plaut Figure 18)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'{self.result_dir}/{PlotTypes.HIDDEN_SIMILARITY}.png', dpi=150)
        if self.result_dir2 is not None:
            plt.savefig(f'{self.result_dir2}/{PlotTypes.HIDDEN_SIMILARITY}.png', dpi=150)


