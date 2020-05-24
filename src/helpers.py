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
 > May 24, 2010
    - Move helper functions from plaut_model.py into this file
    - update string format, filepath, import statements
 > May 08, 2020
    - update create_simulation_folder to create new folder with suffix if folder already exists
      and user does not want to delete the original folder
 > May 03, 2020
    - file created
"""

import os
import shutil

def create_simulation_folder(dir_label):
    # create a new folder for every run
    rootdir = "results/"+dir_label
    try:
        os.mkdir(rootdir)
    except FileExistsError:
        while True:
            replace = input(f"{rootdir} already exists. Replace? (Y/N)")
            if replace in ['Y', 'N', 'y', 'n']:
                break
        if replace in ['Y', 'y']:
            print("Overwriting existing directory...")
            shutil.rmtree(rootdir)
            os.mkdir(rootdir)
        else:
            i = 1
            while True:
                try:
                    new_rootdir = f"{rootdir}_{i}"
                    os.mkdir(new_rootdir)
                    break
                except:
                    i += 1
            rootdir = new_rootdir
            dir_label = f"{dir_label}_{i}"
    for subdir in ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]:
        os.mkdir(rootdir+"/"+subdir)
    return rootdir, dir_label