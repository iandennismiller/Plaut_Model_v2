"""
tsne.py

=== SUMMARY ===
Description     : Create a t-SNE plot
Date Created    : May 03, 2020
Last Updated    : June 16, 2020

=== UPDATE NOTES ===
 > June 13, 2020
    - change filepaths to allow addition of output layer code
    - modify function to choose neighbours (based on same vowel + similar coda)
    - modify plot to only show neighbours, and also colour code neighbours
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
from scipy.spatial.distance import cosine

pd.options.mode.chained_assignment = None

import os
import shutil

filepath = "../results/BASE-S1D1O1-jun13"
pd.set_option('display.max_rows', 10000)


class TSNE_plotter():
    __CODA_SIMILARITY_MAX = 0.01

    def __init__(self, filepath):
        self.rootdir = '/'.join(filepath.split('/')) + '/tsne'
        self.label = filepath.split('/')[-1]

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
                pass

        # Load and process hidden layer and output layer data
        self.hl_df = pd.read_csv(f'{filepath}/warping-dilution-{self.label}-Hidden Layer.csv.gz',
                                 converters={'activation': eval})
        self.hl_df['category'] = self.hl_df['category'].apply(lambda x: self.recategorize(x))
        self.ol_df = pd.read_csv(f'{filepath}/warping-dilution-{self.label}-Output Layer.csv.gz',
                                 converters={'activation': eval})
        self.ol_df['category'] = self.ol_df['category'].apply(lambda x: self.recategorize(x))

        self.tsne = TSNE(n_components=2, perplexity=10, random_state=1)
        # self.hl_df = self.hl_df.astype({'category': 'category'})
        # self.hl_df['label'] = self.hl_df['category'].astype('category').cat.codes
        # self.hl_df['hl_activation'] = self.hl_df['hl_activation'].apply(lambda x: np.array(x))

    def recategorize(self, x):
        if 'ANC' in x:
            return 'ANCHOR'
        elif 'PRO' in x:
            return 'PROBE'
        else:
            return 'BASE'

    def find_neighbours(self, target, epoch, n):
        """
        Find the neighbours that share a common vowel phoneme at the anchor epoch

        Args:
            target (str): the target word to find neighbours of
            epoch (int): the epoch where anchors are added
        """

        base = self.ol_df[(self.ol_df['category'] == 'BASE') & (self.ol_df['epoch'] == epoch)]
        target = self.ol_df[(self.ol_df['orth'].isin(target)) & (self.ol_df['epoch'] == epoch)]
        base['max_vowel'] = base['activation'].apply(lambda x: np.argmax(np.array(x)[23:37]))
        target['max_vowel'] = target['activation'].apply(lambda x: np.argmax(np.array(x)[23:37]))
        base = base[base['max_vowel'].isin(target['max_vowel'].tolist())]
        base['coda_similarity'] = base['activation'].apply(
            lambda x: cosine(x[37:], target['activation'].tolist()[0][37:]))
        base = base[base['coda_similarity'] < TSNE_plotter.__CODA_SIMILARITY_MAX]

        print(target['orth'], base['orth'].tolist())
        return base['orth'].tolist()

    def find_neighbours_old(self, target, epoch, n):
        """
        Find the nearest neighbours of a target word

        Arguments:
            target {list} -- list of orthography of target words to find neighbours of
            n {int} -- number of closest neighbours to find

        Returns:
            neighbours {list} -- list of orthography of the nearest neighours of the target words
        """
        base = self.hl_df[(self.hl_df['category'] == 'BASE') & (self.hl_df['epoch'] == epoch)].reset_index(drop=True)
        base_activations = np.array(base['activation'].tolist())
        neigh = NearestNeighbors(n_neighbors=n)  # +1 b/c it returns the target itself as closest neighbor
        neigh.fit(base_activations)

        target = self.hl_df[(self.hl_df['orth'].isin(target)) & (self.hl_df['epoch'] == epoch)]

        target_activation = np.array(target['activation'].tolist())

        ind = neigh.kneighbors(target_activation, n_neighbors=n, return_distance=False).flatten()

        neighbours = self.hl_df.iloc[ind]['orth'].tolist()

        return neighbours

    def find_distances(self, df, anchor, probes, epoch):
        """
        Find the average distance:
         1. between the anchor and the base vocabulary
         2. between the anchor and the probes
         3. between the probes and the base vocabulary

        Args:
            df (DataFrame): base vocabulary, anchor, and probe data
            anchor (str): orthography of anchor
            probes (list): orthography of the probes
            epoch (int): epoch for calculating distances

        Returns:
            [type]: [description]
        """
        df = df[df['epoch'] == epoch]  # filter for particular epoch

        # find activation of anchor
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

    def label_category(self, row, neighbours1, neighbours2):
        """
        Labels the base vocab words that are neighbours
        Args:
            row (DataFrame row): row of DataFrame corresponding to one word
            neighbours1 (list): first set of neighbours
            neighbours2 (list): second set of neighbours

        Returns:
            Label name for that row
        """
        orth = row['orth']
        if orth in neighbours1 and orth in neighbours2:
            return 'BASE_1+2'
        elif orth in neighbours1:
            return 'BASE_1'
        elif orth in neighbours2:
            return 'BASE_2'
        else:
            return row['category']

    def create_tsne_plot(self, epoch, df, anchor, neighbours1, neighbours2):
        """
        Generates a tsne plot at a particular epoch

        Arguments:
            epoch {int} -- epoch to generate the tsne plot at
            df {DataFrame} -- dataframe of all the orthography and activation data
        """
        df = df[df['epoch'] == epoch]
        df = df.reset_index()
        df['category'] = df.apply(lambda row: self.label_category(row, neighbours1, neighbours2), axis=1)

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

        print('EPOCH 350 NEIGHBOURS')
        neighbours1 = self.find_neighbours(target=anchors, epoch=350, n=50)
        print('EPOCH 700 NEIGHBOURS')
        neighbours2 = self.find_neighbours(target=anchors, epoch=700, n=50)

        neighbours = neighbours1 + neighbours2

        # filter for the neighbours, anchors, and probes
        df = self.hl_df[self.hl_df['orth'].isin(neighbours + anchors + probes)]

        # file to save data
        f = open(self.rootdir + "/tsne.txt", 'a')
        f.write(f'-----------------------------------------\n')
        f.write(f"Anchor: {anchors}\n")
        f.write(f"Probes: {probes}\n")
        f.write("\n")

        # create plot and calculate distances
        for epoch in epochs:
            self.create_tsne_plot(epoch, df, anchors[0], neighbours1, neighbours2)
            anchor_base_dist, anchor_probe_dist, probe_base_dist = self.find_distances(df, anchors, probes, epoch)

            # save distances
            f.write(f"Epoch {epoch}\n")
            f.write(f"--Distance b/w Anchor and Base: {anchor_base_dist}\n")
            f.write(f"--Distance b/w Anchor and Probes: {anchor_probe_dist}\n")
            f.write(f"--Distance b/w Probes and Base: {probe_base_dist}\n")
        f.write(f'-----------------------------------------\n')
        f.close()

        # filepaths = os.listdir(self.rootdir)
        # images = []
        # for path in filepaths:
        #     images.append(imageio.imread(f"{self.rootdir}/{path}"))
        #     imageio.mimsave(f'{self.rootdir}/tsne.gif', images)


t = TSNE_plotter(filepath)

# REGULARS
anchors = ['shing']
probes = ['ging', 'jing', 'ning', 'ting']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['slape']
probes = ['blape', 'clape', 'brape', 'prape']
t.create_tsne_series(anchors=anchors, probes=probes)

# EXCEPTION
anchors = ['nust']
probes = ['chust', 'pust', 'tust', 'drust']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['suff']
probes = ['zuff', 'skuff', 'gluff', 'juff']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['geam']
probes = ['veam', 'frem', 'keam', 'peam']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['fank']
probes = ['pank', 'slank', 'glank', 'brank']
t.create_tsne_series(anchors=anchors, probes=probes)

# AMBIGUOUS
anchors = ['grour']
probes = ['brour', 'drour', 'prour', 'clour']
t.create_tsne_series(anchors=anchors, probes=probes)

anchors = ['plone']
probes = ['blone', 'frone', 'slone', 'smone']
t.create_tsne_series(anchors=anchors, probes=probes)
