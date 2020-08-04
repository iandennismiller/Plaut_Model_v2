"""
plaut_model.py

=== SUMMARY ===
Description     : Code for running a series of simulations
Date Created    : May 03, 2020
Last Updated    : July 26, 2020

=== DETAILED DESCRIPTION ===
 > Given the parameters simulator_config.cfg file, this script will run a series of
   tests with combinations of anchor sets and random seeds

=== UPDATE NOTES ===
 > July 26, 2020
    - add series folder
 > July 18, 2020
    - config filepath and parameter bug fix
    - formatting changes
    - configure preliminary logging
 > July 12, 2020
    - update location of config file
 > May 24, 2020
    - move to root directory, update import statements
    - add functionality to run one single simulation
 > May 08, 2020
    - file created
"""

import argparse
import configparser
import logging
import sys
import yaml
from simulator.simulator import Simulator

parser = argparse.ArgumentParser(description='This script will run a series of \
    simulations with all possible combinations of anchor sets and random seeds given. \
    Note that both the arguments --anchor and --seed are REQUIRED.')

parser.add_argument('sim_type', type=str)
parser.add_argument('-dir', '-d', nargs=1, type=str, metavar='D')
parser.add_argument('-anchor', '-a', type=int, default=0, help='total number of anchor sets', metavar='A')
parser.add_argument('-seed', '-s', nargs='+', type=int, default=[], help='the random seeds to be used', metavar='S')


def write_config_file(anchor, random_seed):
    with open('config/simulator_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

        config['dataset']['anchor_sets'] = anchor
        config['general']['random_seed'] = random_seed

    with open('config/simulator_config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)


def run_simulation(series=False):
    sim = Simulator("config/simulator_config.yaml", series=series)
    sim.train()


if __name__ == "__main__":
    # parse and extract arguments
    args = parser.parse_args()

    assert args.sim_type in ['series', 'single'], "ERROR: Must select sim_type as 'series' or 'single'"

    # set up logging
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler(sys.stdout)
    if args.sim_type == "series":
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s <%(name)s> %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.sim_type == "series":
        max_dilution = args.anchor
        seeds = args.seed

        assert max_dilution > 0, "ERROR: Number of anchor sets must be greater than 0"
        assert len(seeds) > 0, "ERROR: At least one random seed must be specified"

        for seed in seeds:
            # find anchor sets for both upwards and downwards order
            up_sets = list(range(1, max_dilution))
            down_sets = list(range(max_dilution, 0, -1))

            # upwards order
            while len(up_sets) > 0:  # for each dilution level
                logger.info(f"Testing with seed {seed} and anchor sets {up_sets}")
                write_config_file(up_sets, seed)
                run_simulation(series=True)
                up_sets.pop(-1)

            # downwards order
            while len(down_sets) > 0:  # for each dilution level
                logger.info(f"Testing with seed {seed} and anchor sets {down_sets}")
                write_config_file(down_sets, seed)
                run_simulation(series=True)
                down_sets.pop(-1)

    else:
        run_simulation(series=False)
