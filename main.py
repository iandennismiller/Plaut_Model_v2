"""
plaut_model.py

=== SUMMARY ===
Description     : Code for running a series of simulations
Date Created    : May 03, 2020
Last Updated    : May 03, 2020

=== DETAILED DESCRIPTION ===
 > Given the parameters config.cfg file, this script will run a series of
   tests with combinations of anchor sets and random seeds

=== UPDATE NOTES ===
 > May 24, 2020
    - move to root directory, update import statements
    - add functionality to run one single simulation
 > May 08, 2020
    - file created
"""

import argparse
import configparser
from src.simulator import Simulator

parser = argparse.ArgumentParser(description='This script will run a series of \
    simulations with all possible combinations of anchor sets and random seeds given. \
    Note that both the arguments --anchor and --seed are REQUIRED.')

parser.add_argument('sim_type', type=str)

parser.add_argument('-dir', '-d', nargs=1, type=str, metavar='D')

parser.add_argument('-anchor', '-a', type=int, default=0, \
    help='total number of anchor sets', metavar='A')

parser.add_argument('-seed', '-s', nargs='+', type=int, default=[], \
    help='the random seeds to be used', metavar='S')

def write_config_file(anchor, seed):
    config = configparser.ConfigParser()
    config.read('src/config.cfg')

    config['dataset']['anchor_sets'] = str(anchor).strip('[]')
    config['training']['random_seed'] = str(seed)

    with open('src/config.cfg', 'w') as configfile:
        config.write(configfile)
        
def run_simulation():
    sim = Simulator("src/config.cfg")
    sim.train()

if __name__ == "__main__":
    # parse and extract arguments
    args = parser.parse_args() 

    assert args.sim_type in ['series', 'single'], "ERROR: Must select sim_type as 'series' or 'single'"

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
            while len(up_sets) > 0: # for each dilution level
                print(f"Testing with seed {seed} and anchor sets {up_sets}")
                write_config_file(up_sets, seed)
                run_simulation()
                up_sets.pop(-1)
            
            # downwards order
            while len(down_sets) > 0: # for each dilution level 
                print(f"Testing with seed {seed} and anchor sets {down_sets}")
                write_config_file(down_sets, seed)
                run_simulation()
                down_sets.pop(-1)
    
    else:
        run_simulation()
        
        

