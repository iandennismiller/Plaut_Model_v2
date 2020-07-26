"""
helpers.py

=== SUMMARY ===
Description     : Miscellaneous helper functions
Date Created    : May 03, 2020
Last Updated    : July 26, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1:
    - make_folder function renamed to create_simulation_folder, and now forces a folder label to be given

=== UPDATE NOTES ===
 > July 26, 2020
    - add series folder
 > July 19, 2020
    - formatting changes
    - fix too broad except clause
 > May 24, 2020
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


def create_simulation_folder(dir_label, series=False):
    # create a new folder for every run
    if series:
        series_dir = "results/" + dir_label.split('-')[0] + "-" + dir_label.split('-')[-1]
        try:
            os.mkdir(series_dir)
        except FileExistsError:
            pass
        rootdir = series_dir + "/" + dir_label
    else:
        rootdir = "results/" + dir_label

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
                except FileExistsError:
                    i += 1
            rootdir = new_rootdir
            dir_label = f"{dir_label}_{i}"
    for subdir in ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]:
        os.mkdir(rootdir + "/" + subdir)
    return rootdir, dir_label
