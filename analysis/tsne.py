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

filepath = "../results/BASE-S1D2O1-may24/warping-dilution-BASE-S1D2O1-may24-Output Layer.csv.gz"

class TSNE_plotter():
    def __init__(self, filepath):
        self.rootdir = '/'.join(filepath.split('/')[:-1])+'/tsne'
        self.df = pd.read_csv(filepath, converters={'activation': eval}, dtype={'category': 'category'})
        self.df['label'] = self.df['category'].cat.codes
        #self.df['hl_activation'] = self.df['hl_activation'].apply(lambda x: np.array(x))
        
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
        

    def create_tsne_plot(self, epoch):
        df = self.df[self.df['epoch'] == epoch]
        X = df['activation'].to_numpy()
        y = df['label'].to_numpy()
        X = np.vstack(X)
        X = X[:, 23:37]
        X_embedded = TSNE(n_components=2, random_state=1.fit_transform(X)

        # create chart
        plt.figure()
        for cat in enumerate(df['category'].cat.categories):
            ind = np.argwhere(y==cat[0])
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], label=cat[1], s=10)
        plt.legend()
        plt.savefig(f"{self.rootdir}/tsne_epoch{epoch}.png", dpi=200)
        plt.close()

    def create_tsne_gif(self, low, high, interval):
        for epoch in range(low, high+interval, interval):
            self.create_tsne_plot(epoch)
            print(epoch)
        
        filepaths = os.listdir(self.rootdir)
        images = []
        for path in filepaths:
            images.append(imageio.imread(f"{self.rootdir}/{path}"))
            imageio.mimsave(f'{self.rootdir}/tsne.gif', images)


t = TSNE_plotter(filepath)
t.create_tsne_gif(360, 700, 10)