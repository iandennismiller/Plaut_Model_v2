"""
weight_distribution.py

=== SUMMARY ===
Description     : Plot the distribution of weights in each layer of the network
Date Created    : September 20, 2020
Last Updated    : September 20, 2020

=== UPDATE NOTES ===
> September 20, 2020
- file created
"""

import numpy as np
import pandas as pd
import logging
import torch
import shutil
import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from simulator.model import PlautNet
from common.constants import WordTypes, VectorMapping
from common.helpers import *
from tqdm import tqdm
import cv2

pd.set_option('mode.chained_assignment', None)


class WeightDistribution:
    TEMP_PLOTS_FOLDER = 'analysis/weight_distribution'

    def __init__(self, checkpoint_label: str):
        self.logger = logging.getLogger('__main__.' + __name__)
        self.model = PlautNet()
        self.label = checkpoint_label

        self.checkpoint, self.epoch = self.load_checkpoints(self.label)
        self.logger.info('Checkpoints loaded successfully')

        # clear old plots
        shutil.rmtree(WeightDistribution.TEMP_PLOTS_FOLDER)
        os.makedirs(WeightDistribution.TEMP_PLOTS_FOLDER)

    def load_checkpoints(self, label: str):
        """
        Loads all checkpoints for a given simulation label

        Arguments:
            label (str) - simulation label for checkpoints to be loaded

        Returns:
            checkpoint (list) - PyTorch checkpoints, sorted by saved epoch
            epoch (list) - the saved epochs corresponding to each checkpoint
        """
        all_checkpoints = os.listdir('checkpoints')
        checkpoint = [x for x in all_checkpoints if label in x]
        if not checkpoint:
            self.logger.error(f"No checkpoint files exist for given label {label}")
            quit()
        checkpoint = sorted(checkpoint, key=lambda x: int(x.rstrip('.tar').split('_')[-1]))
        epoch = [int(x.rstrip('.tar').split('_')[-1]) for x in checkpoint]
        checkpoint = [torch.load(f"checkpoints/{x}") for x in checkpoint]
        return checkpoint, epoch

    def create_distribution_plots(self):
        """
        Creates the histogram plots for all weights and bias for each layer, for each checkpoint

        Returns:
            None
        """
        for checkpoint, epoch in tqdm(zip(self.checkpoint, self.epoch), total=len(self.epoch),
                                      desc="Creating Weight and Bias Distribution Plots"):
            num_plots = len(checkpoint.keys())
            num_rows = int(np.ceil(num_plots / 2.0))

            sns.set_style('darkgrid')
            fig, axs = plt.subplots(num_rows, 2, figsize=(8, num_rows * 3))

            for key, ax in zip(checkpoint.keys(), axs.reshape(-1)):
                sns.distplot(checkpoint[key], ax=ax)
                ax.set_xlabel('Weight Value' if 'weight' in key else 'Bias Value')
                ax.set_ylabel('Frequency')
                ax.set_title(' '.join(key.split('.')).title())
            plt.suptitle(f'Weight and Bias Distribution - Epoch {epoch}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            plt.savefig(f'{WeightDistribution.TEMP_PLOTS_FOLDER}/epoch_{epoch}.png')

    def create_distribution_video(self, fps: float = 2):
        """
        Combines plots, sorted by epoch, in analysis/weight_distribution folder into video
        Attempts to save video into simulation results folder (otherwise, saved in results folder)

        Arguments:
            fps (float): frame rate for video [default=2]
        """

        # find and sort image filepaths
        images = []
        for f in os.listdir(WeightDistribution.TEMP_PLOTS_FOLDER):
            if f.endswith('.png'):
                images.append(f)
        images = sorted(images, key=lambda x: int(x.rstrip('.png').lstrip('epoch_')))

        # load images
        frame_array = []
        height, width = 0, 0
        for i in images:
            img = cv2.imread(os.path.join(WeightDistribution.TEMP_PLOTS_FOLDER, i))
            height, width, layers = img.shape
            frame_array.append(img)

        # try to find simulation results folder
        if os.path.isdir(f"results/{self.label}"):
            output_folder = f"results/{self.label}"
            self.logger.info(f"Simulation folder found. Saving video in {output_folder}")
        elif os.path.isdir(f"results/{'-'.join(self.label.split('-')[:-2] + [self.label.split('-')[-1]])}"):
            output_folder = f"results/{'-'.join(self.label.split('-')[:-2] + [self.label.split('-')[-1]])}"
            self.logger.info(f"Simulation folder found. Saving video in {output_folder}")
        else:
            output_folder = "results"
            self.logger.info(f"Simulation folder not found. Saving video in {output_folder}")

        # write to video
        out = cv2.VideoWriter(f'{output_folder}/{self.label}_Weight_Distribution.mp4',
                              cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        for i in frame_array:
            out.write(i)
        out.release()
        self.logger.info("Weight distribution and bias distribution video saved successfully")
