"""
density_plots.py

=== SUMMARY ===
Description     : Density plots for hidden layer activations
Date Created    : June 27, 2020
Last Updated    : June 27, 2020

=== UPDATE NOTES ===
 > June 27, 2020
    - file created
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('mode.chained_assignment', None)

filepath = "../results/BASE-S1D1O1-jun27"
label = filepath.split('/')[-1]

categories = ['ANC_REG', 'ANC_EXC', 'ANC_AMB', 'PRO_REG', 'PRO_EXC', 'PRO_AMB']

cat_label_mapping = {'ANC_REG': 'Regular Anchors',
                     'ANC_EXC': 'Exception Anchors',
                     'ANC_AMB': 'Ambiguous Anchors',
                     'PRO_REG': 'Regular Probes',
                     'PRO_EXC': 'Exception Probes',
                     'PRO_AMB': 'Ambiguous Probes'}


def create_density_plot_1(df, epoch, layer):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 8))

    for ax, cat in zip([ax1, ax2, ax3, ax4, ax5, ax6], categories):
        activation_data = df[df['category'] == cat]['activation'].apply(pd.Series)
        sns.distplot(activation_data.to_numpy().reshape(-1), bins=49, kde_kws={'cut': 0, 'bw': 0.025, 'gridsize': 100}, ax=ax)
        ax.set_title(cat_label_mapping[cat])
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
    
    plt.suptitle(f'{layer} Activations - Epoch {epoch}')
    plt.savefig(f'density_plots/{layer} density_plot_with_hist_epoch_{epoch}', dpi=300)
    plt.close()
    
    return None

def create_density_plot_2(df, epoch, layer):
    fig = plt.figure()
    for cat in categories:
        activation_data = df[df['category'] == cat]
        activation_data_new = activation_data['activation'].apply(pd.Series)
        data = activation_data_new.to_numpy().reshape(-1)
        sns.distplot(data,hist=None, kde_kws={'cut': 0, 'bw': 0.025, 'gridsize': 125}, label=cat_label_mapping[cat])

    plt.title(f'{layer} Activations - Epoch {epoch}')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig(f'density_plots/{layer} density_plot_epoch_{epoch}', dpi=300)
    plt.close()
    
    return None


if __name__ == "__main__":
    # load data
    layer = 'Output Layer'
    df = pd.read_csv(f'{filepath}/warping-dilution-{label}-{layer}.csv.gz',
                        converters={'activation': eval})
    df = df[df['category'].isin(categories)]

    for epoch in range(350, 710, 10):
        print(epoch)
        # create_density_plot_1(df[df['epoch'] == epoch], epoch, layer)
        create_density_plot_2(df[df['epoch'] == epoch], epoch, layer)

