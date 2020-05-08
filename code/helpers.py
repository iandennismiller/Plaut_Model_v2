"""
helpers.py

=== SUMMARY ===
Description     : Miscellaneous helper functions
Date Created    : May 03, 2020
Last Updated    : May 03, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1:
    - make_folder function renamed to create_simulation_folder, and now forces a folder label to be given

=== UPDATE NOTES ===
 > May 03, 2020
    - file created
"""

import os
import shutil

def create_simulation_folder(dir_label):
    # create a new folder for every run
    rootdir = "../results/"+dir_label
    try:
        os.mkdir(rootdir)
    except FileExistsError:
        while True:
            delete = input("{} already exists. Delete? (Y/N)".format(rootdir))
            if delete in ['Y', 'N', 'y', 'n']:
                break
        if delete in ['Y', 'y']:
            print("Overwriting existing directory...")
            shutil.rmtree(rootdir)
            os.mkdir(rootdir)
        else:
            print("Simulation Exiting...")
            exit()
    for subdir in ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]:
        os.mkdir(rootdir+"/"+subdir)
    return rootdir