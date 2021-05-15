"""
hidden_similarity_lens.py

=== SUMMARY ===
Description     : Analyze how the hidden representations are organized on the basis of orthographic similarity
                  (see Plaut paper Figure 18)
Date Created    : November 8, 2020
Last Updated    : January 18, 2021

=== UPDATE NOTES ===
 > January 18, 2021
    - create bar plot of correlations at last epoch
 > December 21, 2020
    - minor modifications based on new lens data formatting
 > December 12, 2020
    - make modifications to use lens data over multiple epochs
 > Novemeber 8, 2020
    - file copied from analysis/hidden_similarity.py
    - modifications made to use lens output data
"""

from simulator.model import PlautNet
from common.helpers import *
# from common.constants import PlotTypes
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm

plot_colours = {
    'regular': '#ff7f0e',
    'exception': '#2ca02c',
    'nonword': '#1f77b4'
}
plot_linestyle = {
    'orth_phon_corr': 'solid',
    'orth_hidden_corr': 'dashed',
    'hidden_phon_corr': 'dotted',
    'orth_target_corr': 'dashdot'
}
plot_labels = {
    'orth_phon_corr': 'Orth-Phon',
    'orth_hidden_corr': 'Orth-Hidden',
    'hidden_phon_corr': 'Hidden-Phon',
    'orth_target_corr': 'Orth-Target'
}


class HiddenSimilarityLens:
    def __init__(self, label):
        self.logger = logging.getLogger('__main__.' + __name__)
        self.model = PlautNet()

        self.checkpoint, self.epoch = None, None
        self.checkpoint2, self.epoch2 = None, None
        self.result_dir, self.result_dir2 = None, None

        self.result_dir = f'results/{label}'
        # self.epoch = self.find_epochs() # for old version

    def find_epochs(self):
        files = os.listdir(self.result_dir)
        epoch = set([int(x.split('-')[1]) for x in files if 'activations' in x])
        epoch = sorted(list(epoch))
        print(epoch)
        return epoch

    @staticmethod
    def calculate_correlation_vectors(df):
        dfs = []

        for epoch in tqdm(list(df['epoch'].unique())):
            temp = df[df['epoch'] == epoch].reset_index(drop=True)

            for cat in ['graphemes', 'hidden', 'phonemes']:
                corr_df = temp[cat].apply(pd.Series).T.corr()
                temp[f'{cat}_corr_vector'] = corr_df.values.tolist()
                temp[f'{cat}_corr_vector'] = temp.apply(
                    lambda row: row[f'{cat}_corr_vector'][:row.name] + row[f'{cat}_corr_vector'][row.name+1:], axis=1)
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
        # df['orth_target_corr'] = df.apply(lambda row: np.corrcoef(row['graphemes_corr_vector'],
        #                                                         row['targets_corr_vector'])[0, 1], axis=1)
        return df

    def find_graphemes_hidden_phonemes(self, df):
        df['graphemes'] = df['orth'].apply(lambda x: get_graphemes(x))

        # read results files
        with open(self.result_dir + f"/activations-hidden.txt") as f:
            hidden_results = f.readlines()
        with open(self.result_dir + f"/activations-output.txt") as f:
            output_results = f.readlines()

        # parse hidden layer results file for results
        hidden_data = []
        for line in tqdm(hidden_results, desc='Parsing Hidden Layer Results File: '):
            epoch, orth, hidden = line.split('|')
            hidden = hidden.lstrip('output').split()

            # only keep words that we need for the plot
            if orth not in list(df['orth']):
                continue

            # append to list of hidden data
            hidden_data.append({
                'epoch': int(epoch),
                'orth': orth,
                'hidden': np.array(hidden).astype(float).tolist()
            })
        hidden_df = pd.DataFrame(data=hidden_data)

        # # parse output layer results file for results
        output_data = []
        # target_data = []
        for line in tqdm(output_results, desc='Parsing Outputs Layer Results File: '):
            epoch, orth, output = line.split('|')

            # only keep words that we need for the plot
            if orth not in list(df['orth']):
                continue

            # extract and append data
            if 'output' in line:
                output = output.lstrip('output').split()
                output_data.append({
                    'epoch': int(epoch),
                    'orth': orth,
                    'phonemes': np.array(output).astype(float).tolist()
                })
            # elif 'target' in line:
            #     output = output.lstrip('target').split()
            #     target_data.append({
            #         'epoch': int(epoch),
            #         'orth': orth,
            #         'targets': np.array(output).astype(float).tolist()
            #     })
            else:
                continue

        output_df = pd.DataFrame(data=output_data)
        # target_df = pd.DataFrame(data=target_data)
        # output_df = output_df.merge(target_df, on=['orth', 'epoch'])

        print(hidden_df)
        result_df = pd.merge(hidden_df, output_df, on=['orth', 'epoch'])
        result_df = pd.merge(result_df, df, on='orth')

        self.epoch = result_df['epoch'].unique()

        return result_df

    # def find_graphemes_hidden_phonemes(self, df, epoch):
    #     df['graphemes'] = df['orth'].apply(lambda x: get_graphemes(x))
    #
    #     dfs = []
    #     for e in epoch:
    #         with open(self.result_dir + f"/recurrent-{e}-activations-hidden.txt") as f:
    #             hidden_results = f.readlines()
    #         with open(self.result_dir + f"/recurrent-{e}-activations-output.txt") as f:
    #             output_results = f.readlines()
    #
    #         hidden_data = []
    #         for line in hidden_results:
    #             orth = line.split('|')[0]
    #             # orth = line.split('|')[0].split('_')[1]
    #             if orth not in list(df['orth']):
    #                 continue
    #
    #             hidden = line.split('|')[1].lstrip('output').split()
    #             hidden_data.append({
    #                 'orth': orth,
    #                 'hidden': np.array(hidden).astype(float).tolist()
    #             })
    #         hidden_df = pd.DataFrame(data=hidden_data)
    #
    #         output_data = []
    #         for line in output_results:
    #             if 'target' in line:
    #                 continue
    #             # orth = line.split('|')[0].split('_')[1]
    #             orth = line.split('|')[0]
    #             if orth not in list(df['orth']):
    #                 continue
    #             output = line.split('|')[1].lstrip('output').split()
    #             output_data.append({
    #                 'orth': orth,
    #                 'phonemes': np.array(output).astype(float).tolist()
    #             })
    #         output_df = pd.DataFrame(data=output_data)
    #         pd.set_option('display.max_columns', None)
    #         results_df = pd.merge(hidden_df, output_df, on='orth')
    #         results_df['epoch'] = e
    #         results_df = results_df.merge(df, on='orth')
    #
    #         dfs.append(results_df)
    #
    #         # self.model.load_state_dict(c)
    #         # self.model.eval()
    #         # hidden, phon = self.model(torch.tensor(df['graphemes'], dtype=torch.float))
    #         # dfs.append(pd.DataFrame(data={'orth': df['orth'],
    #         #                               'epoch': [e] * len(df),
    #         #                               'graphemes': df['graphemes'],
    #         #                               'hidden': hidden.tolist(),
    #         #                               'phonemes': phon.tolist()}))
    #
    #     return pd.concat(dfs).reset_index(drop=True)

    def create_plots(self, dataset_filename):
        test_dataset = pd.read_csv(f"dataset/{dataset_filename}")
        regulars2, exceptions2, nonwords2 = None, None, None

        # extract orthology of words
        regulars = test_dataset['Regular Inconsistent'].to_frame().rename(columns={'Regular Inconsistent': 'orth'})
        exceptions = test_dataset['Exception'].to_frame().rename(columns={'Exception': 'orth'})
        nonwords = test_dataset['Nonword'].to_frame().rename(columns={'Nonword': 'orth'})

        # find graphemes, hidden, phonemes
        self.logger.info('Extracting Regulars...')
        regulars = self.find_graphemes_hidden_phonemes(regulars)
        self.logger.info('Extracting Exceptions...')
        exceptions = self.find_graphemes_hidden_phonemes(exceptions)
        self.logger.info('Extracting Nonwords...')
        nonwords = self.find_graphemes_hidden_phonemes(nonwords)

        # calculate correlation for same layer between words
        self.logger.info('Calculating intra-layer correlation...')
        regulars = self.calculate_correlation_vectors(regulars)
        exceptions = self.calculate_correlation_vectors(exceptions)
        nonwords = self.calculate_correlation_vectors(nonwords)

        # calculate correlation between layers for same word
        self.logger.info('Calculating inter-layer correlation...')
        regulars = self.calculate_cross_layer_correlation(regulars)
        exceptions = self.calculate_cross_layer_correlation(exceptions)
        nonwords = self.calculate_cross_layer_correlation(nonwords)

        # if self.checkpoint2:
        #     regulars2 = test_dataset['Regular Inconsistent'].to_frame().rename(columns={'Regular Inconsistent': 'orth'})
        #     exceptions2 = test_dataset['Exception'].to_frame().rename(columns={'Exception': 'orth'})
        #     nonwords2 = test_dataset['Nonword'].to_frame().rename(columns={'Nonword': 'orth'})
        #
        #     regulars2 = self.find_graphemes_hidden_phonemes(regulars2,  self.epoch2)
        #     exceptions2 = self.find_graphemes_hidden_phonemes(exceptions2, self.epoch2)
        #     nonwords2 = self.find_graphemes_hidden_phonemes(nonwords2, self.epoch2)
        #
        #     regulars2 = self.calculate_correlation_vectors(regulars2)
        #     exceptions2 = self.calculate_correlation_vectors(exceptions2)
        #     nonwords2 = self.calculate_correlation_vectors(nonwords2)
        #
        #     regulars2 = self.calculate_cross_layer_correlation(regulars2)
        #     exceptions2 = self.calculate_cross_layer_correlation(exceptions2)
        #     nonwords2 = self.calculate_cross_layer_correlation(nonwords2)

        # average and plot
        cols = ['orth_phon_corr', 'orth_hidden_corr', 'hidden_phon_corr']
        max_epoch = regulars['epoch'].max()
        num_samples = len(regulars[regulars['epoch'] == max_epoch])
        regulars_line = regulars[['epoch']+cols].groupby(by='epoch').mean()
        exceptions_line = exceptions[['epoch']+cols].groupby(by='epoch').mean()
        nonwords_line = nonwords[['epoch']+cols].groupby(by='epoch').mean()
        regulars_bar = regulars[regulars['epoch'] == max_epoch][cols].agg(['mean', 'std'])
        exceptions_bar = exceptions[exceptions['epoch'] == max_epoch][cols].agg(['mean', 'std'])
        nonwords_bar = nonwords[nonwords['epoch'] == max_epoch][cols].agg(['mean', 'std'])

        bar_mean = pd.concat([nonwords_bar.loc[['mean']].rename(index={'mean': 'Nonwords'}),
                              regulars_bar.loc[['mean']].rename(index={'mean': 'Regulars'}),
                              exceptions_bar.loc[['mean']].rename(index={'mean': 'Exceptions'})])
        bar_mean = bar_mean.rename(columns=plot_labels)

        bar_error = pd.concat([nonwords_bar.loc[['std']].rename(index={'std': 'Nonwords'}),
                               regulars_bar.loc[['std']].rename(index={'std': 'Regulars'}),
                               exceptions_bar.loc[['std']].rename(index={'std': 'Exceptions'})])
        print(num_samples)
        bar_error = bar_error.rename(columns=plot_labels) / np.sqrt(num_samples)

        # if self.checkpoint2:
        #     regulars2 = regulars2[cols].groupby(by='epoch').mean()
        #     exceptions2 = exceptions2[cols].groupby(by='epoch').mean()
        #     nonwords2 = nonwords2[cols].groupby(by='epoch').mean()
        #     regulars = pd.concat([regulars, regulars2]).groupby(by='epoch').mean()
        #     exceptions = pd.concat([exceptions, exceptions2]).groupby(by='epoch').mean()
        #     nonwords = pd.concat([nonwords, nonwords2]).groupby(by='epoch').mean()

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(9, 6))
        for i in cols:
            regulars_line[i].plot.line(ax=ax, color=plot_colours['regular'], marker='.',
                                       linestyle=plot_linestyle[i], label=f'{plot_labels[i]} | Regular')
            exceptions_line[i].plot.line(ax=ax, color=plot_colours['exception'], marker='.',
                                         linestyle=plot_linestyle[i], label=f'{plot_labels[i]} | Exception')
            nonwords_line[i].plot.line(ax=ax, color=plot_colours['nonword'], marker='.',
                                       linestyle=plot_linestyle[i], label=f'{plot_labels[i]} | Nonword')

        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.ylim(0.5, 1)
        plt.suptitle('Similarity Correlations (Plaut Figure 18)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=len(cols))
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'{self.result_dir}/hidden_similarity.png', dpi=150)

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_mean.T.plot.bar(rot=0, yerr=bar_error.T, capsize=4, ax=ax)
        plt.suptitle('Similarity Correlations (Plaut Figure 18)')
        plt.ylabel('Correlation')
        plt.ylim(0.5, 1)
        plt.legend()
        plt.minorticks_on()
        plt.grid(b=True, which='minor', axis='y', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'{self.result_dir}/hidden_similarity_bar.png', dpi=150)

        # if self.result_dir2 is not None:
        #     plt.savefig(f'{self.result_dir2}/{PlotTypes.HIDDEN_SIMILARITY}.png', dpi=150)


