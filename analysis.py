"""
analysis.py

=== SUMMARY ===
Description     : Code for running an analysis on results data
Date Created    : July 28, 2020
Last Updated    : July 28, 2020

=== DETAILED DESCRIPTION ===
 > TBD

=== UPDATE NOTES ===
 > July 28, 2020
    - file created
"""

import argparse
import logging
import os
import sys

from analysis.density_plots import DensityPlots

parser = argparse.ArgumentParser(description='This script will run the specified analysis script')

parser.add_argument('analysis_type', type=str)
parser.add_argument('results_dir', type=str)

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

    if not os.path.isdir(f"results/{args.results_dir}"):
        logger.error(f"Folder results/{args.results_dir} does not exist")
        quit()

    if args.analysis_type in ['density_plots', 'dp']:
        # dp = DensityPlots(folder=f"results/{args.results_dir}", anchor=True, probe=True)
        dp = DensityPlots(folder=f"results/{args.results_dir}", plaut=True)
        dp.create_hl_plots()
        # dp.create_ol_plots()
