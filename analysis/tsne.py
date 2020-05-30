"""
tsne.py

=== SUMMARY ===
Description     : Create a t-SNE plot
Date Created    : May 03, 2020
Last Updated    : May 03, 2020

=== UPDATE NOTES ===
 > May 24, 2010
    - file created
"""

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import imageio

filepath = "../results/BASE-S1D1O1-may30/warping-dilution-BASE-S1D1O1-may30-Hidden Layer.csv.gz"
#anchors = ['blome']
#probes = ['stome', 'shome', 'grome', 'prome']


class TSNE_plotter():
    def __init__(self, filepath):
        self.rootdir = '/'.join(filepath.split('/')[:-1])+'/tsne'

        # create folder for tsne
        try:
            os.mkdir(self.rootdir)
        except FileExistsError:
            while True:
                replace = input(f"{self.rootdir} already exists. Replace? (Y/N)")
                if replace in ['Y', 'N', 'y', 'n']:
                    break
            if replace in ['Y', 'y']:
                print("Overwriting existing directory...")
                shutil.rmtree(self.rootdir)
                os.mkdir(self.rootdir)
            else:
                print("Exiting...")
                exit()
        
        # Load and process data
        self.df = pd.read_csv(filepath, converters={'activation': eval})
        self.df['category'] = self.df['category'].apply(lambda x: self.recategorize(x))
        
        self.tsne = TSNE(n_components=2, perplexity=100, random_state=1)
        #self.df = self.df.astype({'category': 'category'})
        #self.df['label'] = self.df['category'].astype('category').cat.codes
        #self.df['hl_activation'] = self.df['hl_activation'].apply(lambda x: np.array(x))
        
    def recategorize(self, x):
        if 'ANC' in x:
            return 'ANCHOR'
        elif 'PRO' in x:
            return 'PROBE'
        else:
            return 'BASE'
        
    def create_tsne_plot(self, epoch, df):
        df = df[df['epoch'] == epoch]
        df = df.reset_index()
        X = df['activation'].to_numpy()
        #y = df['label'].to_numpy()
        X = np.vstack(X)
        X_embedded = self.tsne.fit_transform(X)

        # create chart
        plt.figure()
        for cat in df['category'].unique():
            ind = df.index[df['category'] == cat].tolist()
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], label=cat, s=5)
        plt.legend()
        plt.savefig(f"{self.rootdir}/tsne_epoch{epoch}.png", dpi=200)
        plt.close()

    def create_tsne_gif(self, anchors, probes):
        epochs = [350, 700]#self.df['epoch'].unique()
        # filter out extra anchors and probes
        df = self.df[(self.df['category'] == 'BASE') | ((self.df['category'] == 'ANCHOR') & (self.df['orth'].isin(anchors))) | ((self.df['category'] == 'PROBE') & (self.df['orth'].isin(probes)))]

        for epoch in epochs:
            self.create_tsne_plot(epoch, df)
        '''
        filepaths = os.listdir(self.rootdir)
        images = []
        for path in filepaths:
            images.append(imageio.imread(f"{self.rootdir}/{path}"))
            imageio.mimsave(f'{self.rootdir}/tsne.gif', images)
        '''


t = TSNE_plotter(filepath)
t.create_tsne_gif(anchors, probes)