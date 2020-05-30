# Plaut_Model_v2

## Setup Requirements
Please refer to `requirements.txt`.

## General Information
There are several directories **required** for full functionality:
 1. `checkpoints` This directory stores any created checkpoints for future use. This directory can be omitted if no checkpoints will be used or created.
 2. `src` This directory contains the code and also the configuration file.
 3. `dataset` This directory contains the Plaut, anchor, and probe datasets in `.csv` format.
 4. `results` This directory stores the simulation results folders, which contain the plots and `.csv.gz` files created with each simulation.


## How to Use
1. The configuration file for simulations is located in `code/config.cfg`. This file contains all user-settable parameters. The description of each parameter is available in `code/README.md`. This file must **not** be renamed nor moved in order for the simulation to run successfully. With each simulation, a copy of the `config.cfg` file will be placed in the simulation results folder.
2. There are two ways to run a simulation. Both methods below must be done in the `code` directory
    - Using `simulator.py`
        - Executing `python3 simulator.py` will run a single simulation based on the settings in `config.cfg`
    - Using `plaut_model.py`
        - Executing `python3 plaut_model.py -anchor A -seed S [S ...]` can run a series of simulations, with all settings based on `config.cfg` **except** for `anchor_sets` and `random_seed`, which are defined below
        - `A` is the total number of anchor sets. In the Anchor Dataset file, the anchor sets must be numbered `1, 2, ..., A`.
        - `[S ...]` is a series of random seeds to be used
        - For each seed value given, a "upward" series of simulations will be run with `anchor_sets = 1` up to `anchor_sets = 1, ..., A-1`, and a "downward" series of `anchor_sets = A, ..., 1` down to `anchor_sets = A`
        - Recommended: `python3 plaut_model.py -anchor 3 -seed 1 2`
3. A folder containing simulation results will be created inside the `results` directory. This simulation folder will be named with the format `<label>-S<seed>D<dilution>O<order>-<date>`. Inside the simulation folder will be loss and accuracy plots, a copy of the configuration file, as well as `.csv.gz` file containing accuracy information for each word at specified intervals during the simulation.