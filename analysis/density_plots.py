"""
density_plots.py

=== SUMMARY ===
Description     : Density plots for hidden layer activations
Date Created    : June 27, 2020
Last Updated    : August 5, 2020

=== UPDATE NOTES ===
 > August 5, 2020
    - function added for output layer inputs
 > July 31, 2020
    - add function for combining images into video
 > July 27, 2020
    - update functions, and put into class
 > June 27, 2020
    - file created
"""

import numpy as np
import pandas as pd
import logging
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from common.constants import WordTypes, VectorMapping
from common.helpers import *
from tqdm import tqdm
import cv2

pd.set_option('mode.chained_assignment', None)


class DensityPlots:
    def __init__(self, results_folder):
        self.logger = logging.getLogger('__main__.' + __name__)
        self.results_dir = results_folder
        self.label = results_folder.split('/')[-1]

        # CREATE FOLDER FOR DENSITY PLOTS IF NOT ALREADY CREATED
        self.dp_folder = f"{self.results_dir}/density_plots"
        if not os.path.isdir(self.dp_folder):
            self.dp_folder = create_analysis_folder(self.results_dir, 'density_plots')

        self.logger.info(f"Density plots will be stored in {self.dp_folder}")

    @staticmethod
    def get_categories(plaut, anchor, probe):
        categories = []
        if plaut:
            categories += WordTypes.plaut_types
        if anchor:
            categories += WordTypes.anchor_types
        if probe:
            categories += WordTypes.probe_types
        return categories

    @staticmethod
    def get_folder_name(plaut, anchor, probe):
        name = ""
        if plaut:
            name += "_plaut"
        if anchor:
            name += "_anchor"
        if probe:
            name += "_probe"
        return name

    def create_hl_activation_plots(self, plaut=False, anchor=False, probe=False):
        # CHECK WHETHER DATA FILE EXISTS
        filepath = f'{self.results_dir}/warping-dilution-{self.label}-Hidden Layer Activations.csv.gz'
        if not os.path.isfile(filepath):
            self.logger.error('No data found for hidden layer activations in given folder')
            return None

        categories = self.get_categories(plaut, anchor, probe)
        folder_name = 'hidden_activations' + self.get_folder_name(plaut, anchor, probe)

        # CREATE FOLDER FOR OUTPUTS
        output_dir = create_analysis_folder(self.dp_folder, folder_name)

        # LOAD DATA
        self.logger.info('Loading hidden layer activation data')
        df = pd.read_csv(filepath, converters={'activation': eval})
        df = df[df['category'].isin(categories)]

        # CREATE DENSITY PLOTS
        y_min, y_max = None, None
        t = tqdm(sorted(df['epoch'].unique(), reverse=True))

        for epoch in t:
            t.set_description(f"[Hidden Layer Activation - Epoch {epoch:3d}]")
            epoch_df = df[df['epoch'] == epoch]
            plt.figure()
            for cat in categories:
                label = self.get_category_label(cat)

                activation_data = epoch_df[epoch_df['category'] == cat]['activation'].apply(pd.Series)
                data = activation_data.to_numpy().reshape(-1)
                sns.distplot(data, hist=None, kde_kws={'bw': 0.025, 'gridsize': 150}, label=label)

            if y_min is None:
                _, _, y_min, y_max = plt.axis()
            else:
                plt.ylim(y_min, y_max)

            plt.title(f'Hidden Layer Activations - Epoch {epoch}')
            plt.legend()
            plt.savefig(f'{output_dir}/{folder_name}_{epoch}', dpi=300)
            plt.close()

        self.combine_plots_as_video(output_dir, folder_name)
        self.logger.info("Completed all required density plots for hidden layer activations")

    def create_ol_activation_plots(self, plaut=False, anchor=False, probe=False):
        # CHECK WHETHER DATA FILE EXISTS
        filepath = f'{self.results_dir}/warping-dilution-{self.label}-Output Layer Activations.csv.gz'
        if not os.path.isfile(filepath):
            logging.error('No data found for output layer activations in given folder')
            return None

        categories = self.get_categories(plaut, anchor, probe)
        folder_name = 'output_activations' + self.get_folder_name(plaut, anchor, probe)

        # CREATE FOLDER FOR OUTPUTS
        output_dir = create_analysis_folder(self.dp_folder, folder_name)

        # LOAD DATA
        self.logger.info('Loading output layer activation data')
        df = pd.read_csv(filepath, converters={'activation': eval})
        df = df[df['category'].isin(categories)]

        num_onset = len(VectorMapping.phoneme_onset)
        num_vowel = len(VectorMapping.phoneme_vowel)

        # CREATE DENSITY PLOTS
        t = tqdm(df['epoch'].unique())
        for epoch in t:
            t.set_description(f"[Output Layer Activation - Epoch {epoch:3d}]")
            epoch_df = df[df['epoch'] == epoch]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharex='all', sharey='all')

            labels = []
            for cat in categories:
                label = self.get_category_label(cat)
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
            plt.savefig(f'{output_dir}/{folder_name}_{epoch}', dpi=300)
            plt.close()

        self.combine_plots_as_video(output_dir, folder_name)
        self.logger.info("Completed all required density plots for output layer activations")

    def create_ol_input_plots(self, plaut=False, anchor=True, probe=False):
        hl_activations_filepath = f'{self.results_dir}/warping-dilution-{self.label}-Hidden Layer Activations.csv.gz'
        ol_activations_filepath = f'{self.results_dir}/warping-dilution-{self.label}-Output Layer Activations.csv.gz'
        weights_filepath = f'{self.results_dir}/warping-dilution-{self.label}-Model Weights.pkl'
        # CHECK WHETHER FILES EXIST
        if not os.path.isfile(hl_activations_filepath):
            self.logger.error('No data found for hidden layer activations in given folder')
            return None
        if not os.path.isfile(ol_activations_filepath):
            self.logger.error('No data found for output layer activations in given folder')
            return None
        if not os.path.isfile(weights_filepath):
            self.logger.error('No data found for weights in given folder')
            return None

        categories = self.get_categories(plaut, anchor, probe)
        folder_name = 'output_layer_inputs' + self.get_folder_name(plaut, anchor, probe)

        # CREATE FOLDER FOR OUTPUTS
        output_dir = create_analysis_folder(self.dp_folder, folder_name)

        # LOAD DATA
        self.logger.info('Loading hidden layer activation data')
        hl_df = pd.read_csv(hl_activations_filepath, converters={'activation': eval})
        hl_df = hl_df[hl_df['category'].isin(categories)]
        self.logger.info('Loading output layer activation data')
        ol_df = pd.read_csv(ol_activations_filepath, converters={'activation': eval})
        ol_df = ol_df[ol_df['category'].isin(categories)]
        self.logger.info('Loading weights')
        weights_df = pd.read_pickle(weights_filepath)

        activations = pd.merge(hl_df, ol_df, on=['epoch', 'orth', 'category'], suffixes=['_hl', '_ol'])
        epochs = activations['epoch'].unique()
        words = activations['orth'].unique()

        weight_result = {cat: {e: {'regularized': [], 'training consistent': []} for e in epochs} for cat in categories}
        bias_result = {cat: {e: {'regularized': [], 'training consistent': []} for e in epochs} for cat in categories}

        for word in tqdm(words, desc='Calculating Weight and Bias Inputs'):
            temp = activations[activations['orth'] == word].sort_values(by='epoch').reset_index(drop=True)

            num_onsets = len(VectorMapping.phoneme_onset)
            num_vowels = len(VectorMapping.phoneme_vowel)
            r_vowel = np.argmax(temp.iloc[0]['activation_ol'][num_onsets:(num_onsets+num_vowels)]) + num_onsets
            tc_vowel = np.argmax(temp.iloc[-1]['activation_ol'][num_onsets:(num_onsets+num_vowels)]) + num_onsets

            for i, row in temp.iterrows():
                e = row['epoch']
                cat = row['category']
                hl_activations = row['activation_hl']
                layer2_weights = weights_df.loc[e, 'weights']['layer2.weight']
                layer2_bias = weights_df.loc[e, 'weights']['layer2.bias']
                print(layer2_weights)

                # regularized weight inputs
                r_weights = np.multiply(hl_activations, layer2_weights[r_vowel, :])
                weight_result[cat][e]['regularized'].append(r_weights.numpy())

                # training consistent weight inputs
                tc_weights = np.multiply(hl_activations, layer2_weights[tc_vowel, :])
                weight_result[cat][e]['training consistent'].append(tc_weights.numpy())

                # regularized bias
                bias_result[cat][e]['regularized'].append(layer2_bias[r_vowel].numpy())
                bias_result[cat][e]['training consistent'].append(layer2_bias[tc_vowel].numpy())

        for e in tqdm(epochs, desc='Creating Density Plots of Weight Inputs'):
            fig, axs = plt.subplots(1, len(categories), figsize=(12, 6))
            for cat, ax in zip(categories, axs):
                for vowel_type in ['regularized', 'training consistent']:
                    sns.distplot(np.hstack(weight_result[cat][e][vowel_type]), hist=None,
                                 kde_kws={'bw': 0.05, 'gridsize': 150}, label=f'{vowel_type.title()} Vowel', ax=ax)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
                ax.set_title(self.get_category_label(cat))

            plt.suptitle(f'Output Layer Weight Inputs - Epoch {e}')
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            plt.savefig(f'{output_dir}/{folder_name}_weight_{e}.png', dpi=300)
            plt.close()

        self.logger.info('Merging density plots for weight inputs as video')
        self.combine_plots_as_video(output_dir, folder_name, suffix='Weight Inputs')
        for f in os.listdir(output_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(output_dir, f))

        for e in tqdm(epochs, desc='Creating Density Plots of Bias Inputs'):
            fig, axs = plt.subplots(1, len(categories), figsize=(12, 6))
            for cat, ax in zip(categories, axs):
                for vowel_type in ['regularized', 'training consistent']:
                    sns.distplot(np.hstack(bias_result[cat][e][vowel_type]), hist=None,
                                 kde_kws={'bw': 0.05, 'gridsize': 150}, label=f'{vowel_type.title()} Vowel', ax=ax)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
                ax.set_title(self.get_category_label(cat))

            plt.suptitle(f'Output Layer Bias Inputs - Epoch {e}')
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            plt.savefig(f'{output_dir}/{folder_name}_bias_{e}.png', dpi=300)
            plt.close()

        self.logger.info('Merging density plots for weight inputs as video')
        self.combine_plots_as_video(output_dir, folder_name, suffix='Bias Inputs')
        for f in os.listdir(output_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(output_dir, f))

        self.logger.info("Completed density plots for output layer inputs")

    @staticmethod
    def combine_plots_as_video(output_dir, name, suffix="", fps=2):
        # FIND AND SORT IMAGE FILEPATHS
        images = []
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                images.append(f)
        images = sorted(images, key=lambda x: int(x.split('_')[-1][:-4]))

        # LOAD IMAGES
        frame_array = []
        width, height = 0, 0
        for i in images:
            img = cv2.imread(os.path.join(output_dir, i))
            height, width, layers = img.shape
            frame_array.append(img)

        # WRITE TO VIDEO
        out = cv2.VideoWriter(f'{output_dir}/{name}_{suffix}.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        for i in frame_array:
            out.write(i)
        out.release()

    @staticmethod
    def get_category_label(cat):
        if cat in WordTypes.anchor_types:
            return WordTypes.anchor_mapping[cat]
        elif cat in WordTypes.probe_types:
            return WordTypes.probe_mapping[cat]
        else:
            return cat
