"""
tsne.py

=== SUMMARY ===
Description     : Create a t-SNE plot
Date Created    : May 03, 2020
Last Updated    : May 03, 2020

=== UPDATE NOTES ===
 > June 6, 2020
    - add function to find the closest neighbours
 > May 30, 2020
    - added filter for probes and anchors
    - modified for loop for scatter plot and remove converting 'category' column as category dtype
 > May 24, 2020
    - file created
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
pd.options.mode.chained_assignment = None

import os
import shutil

filepath = "../results/BASE-S1D1O1-may30/warping-dilution-BASE-S1D1O1-may30-Hidden Layer.csv.gz"

class TSNE_plotter():
    def __init__(self, filepath):
        self.rootdir = '/'.join(filepath.split('/')[:-1])+'/tsne'

        # create folder for tsne
        
        try:
            os.mkdir(self.rootdir)
        except FileExistsError:
            pass
            while True:
                replace = input(f"{self.rootdir} already exists. Replace? (Y/N)")
                if replace in ['Y', 'N', 'y', 'n']:
                    break
            if replace in ['Y', 'y']:
                print("Overwriting existing directory...")
                shutil.rmtree(self.rootdir)
                os.mkdir(self.rootdir)
            else:
                pass
        
        # Load and process data
        self.df = pd.read_csv(filepath, converters={'activation': eval})
        self.df['category'] = self.df['category'].apply(lambda x: self.recategorize(x))
        
        self.tsne = TSNE(n_components=2, random_state=1)
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
    
    def find_neighbours(self, target, epoch, n):
        """
        Find the nearest neighbours of a target word

        Arguments:
            target {list} -- list of orthography of target words to find neighbours of
            n {int} -- number of closest neighbours to find

        Returns:
            neighbours {list} -- list of orthography of the nearest neighours of the target words
        """
        base = self.df[(self.df['category'] == 'BASE') & (self.df['epoch'] == epoch)].reset_index(drop=True)
        base_activations = np.array(base['activation'].tolist())
        neigh = NearestNeighbors(n_neighbors=n) # +1 b/c it returns the target itself as closest neighbor
        neigh.fit(base_activations)
        
        target = self.df[(self.df['orth'].isin(target)) & (self.df['epoch'] == epoch)]
        
        target_activation = np.array(target['activation'].tolist())
        
        ind = neigh.kneighbors(target_activation, n_neighbors=n, return_distance=False).flatten()

        neighbours = self.df.iloc[ind]['orth'].tolist()
        
        return neighbours
    
    def find_distances(self, df, anchor, probes, epoch):
        df = df[df['epoch'] == epoch]

        anchor_activation = np.array(df[df['orth'].isin(anchor)]['activation'].tolist()).flatten()
        
        df['dist'] = df['activation'].apply(lambda x: np.linalg.norm(anchor_activation - np.array(x)))
        
        # distance w/ anchors
        anchor_base_dist = df[df['category'] == 'BASE']['dist'].mean()
        anchor_probe_dist = df[df['category'] == 'PROBE']['dist'].mean()

        # for probes
        probe_base_dist = 0
        for p in probes:
            p_activation = np.array(df[df['orth'] == p]['activation'].tolist()).flatten()
            df['dist'] = df['activation'].apply(lambda x: np.linalg.norm(p_activation - np.array(x)))
            probe_base_dist += df[df['category'] == 'BASE']['dist'].mean()

        return anchor_base_dist, anchor_probe_dist, probe_base_dist / len(probes)
        
    def create_tsne_plot(self, epoch, df, anchor):
        """
        Generates a tsne plot at a particular epoch

        Arguments:
            epoch {int} -- epoch to generate the tsne plot at
            df {DataFrame} -- dataframe of all the orthography and activation data
        """
        df = df[df['epoch'] == epoch]
        df = df.reset_index()
        
        X = df['activation'].to_numpy()
        X = np.vstack(X)
        X_embedded = self.tsne.fit_transform(X)

        # create chart
        plt.figure()
        for cat in df['category'].unique():
            ind = df.index[df['category'] == cat].tolist()
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], label=cat, s=5)
        plt.legend()
        plt.savefig(f"{self.rootdir}/tsne_epoch{epoch}_{anchor}.png", dpi=200)
        plt.close()

    def create_tsne_series(self, anchors, probes):
        epochs = [350, 700]

        neighbours = self.find_neighbours(target=anchors, epoch=350, n=50)

        # filter for the neighbours, anchors, and probes
        df = self.df[self.df['orth'].isin(neighbours+anchors+probes)]

        f = open(self.rootdir+"/tsne.txt", 'a')
        f.write(f'-----------------------------------------\n')
        f.write(f"Anchor: {anchors}\n")
        f.write(f"Probes: {probes}\n")
        f.write("\n")
        
        for epoch in epochs:
            self.create_tsne_plot(epoch, df, anchors[0])
            anchor_base_dist, anchor_probe_dist, probe_base_dist = self.find_distances(df, anchors, probes, epoch)
            f.write(f"Epoch {epoch}\n")
            f.write(f"--Distance b/w Anchor and Base: {anchor_base_dist}\n")
            f.write(f"--Distance b/w Anchor and Probes: {anchor_probe_dist}\n")
            f.write(f"--Distance b/w Probes and Base: {probe_base_dist}\n")
        f.write(f'-----------------------------------------\n')
        f.close()
        
        '''
        filepaths = os.listdir(self.rootdir)
        images = []
        for path in filepaths:
            images.append(imageio.imread(f"{self.rootdir}/{path}"))
            imageio.mimsave(f'{self.rootdir}/tsne.gif', images)
        '''

t = TSNE_plotter(filepath)

anchors = ['shing']
probes=['ging', 'jing', 'ning', 'ting']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['slape']
probes=['blape', 'clape', 'brape', 'prape']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['nust']
probes=['chust', 'pust', 'tust', 'drust']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['suff']
probes=['zuff', 'skuff', 'gluff', 'juff']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['grour']
probes=['brour', 'drour', 'prour', 'clour']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['plone']
probes=['blone', 'frone', 'slone', 'smone']
t.create_tsne_series(anchors=anchors, probes=probes)
