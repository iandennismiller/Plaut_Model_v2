"""
analysis.py

=== SUMMARY ===
Description     : Code for running an analysis on results data
Date Created    : July 28, 2020
Last Updated    : August 5, 2020

=== DETAILED DESCRIPTION ===
 > TBD

=== UPDATE NOTES ===
 > August 5, 2020
    - script modified based on changes from density plots
 > July 28, 2020
    - file created
"""

import argparse
import logging
import os
import sys

from analysis.density_plots import DensityPlots
from analysis.hidden_similarity import HiddenSimilarity

parser = argparse.ArgumentParser(description='This script will run the specified analysis script')

parser.add_argument('analysis_type', type=str)
parser.add_argument('-r', '--results_dir', type=str)
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-c2', '--checkpoint2', type=str)
parser.add_argument('-d', '--dataset', type=str)

if __name__ == "__main__":
    # set up logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s <%(name)s> %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # parse and extract arguments
    args = parser.parse_args()

    if args.analysis_type in ['density_plots', 'dp']:
        if not os.path.isdir(f"results/{args.results_dir}"):
            logger.error(f"Folder results/{args.results_dir} does not exist or is not given")
            quit()
        dp = DensityPlots(results_folder=f"results/{args.results_dir}")
        dp.create_hl_activation_plots(plaut=True, anchor=False, probe=False)
        dp.create_hl_activation_plots(plaut=False, anchor=True, probe=True)
        dp.create_ol_activation_plots(plaut=False, anchor=True, probe=True)
        dp.create_ol_input_plots(plaut=False, anchor=True, probe=False)

    elif args.analysis_type in ['hidden_similarity', 'hs']:
        hs = HiddenSimilarity(args.checkpoint, args.checkpoint2)
        hs.create_plots(args.dataset)
