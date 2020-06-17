"""
stress.py

=== SUMMARY ===
Description     : Script stress calculations on data
Date Created    : June 16, 2020
Last Updated    : June 16, 2020

=== UPDATE NOTES ===
 > June 13, 2020
    - file created
"""

import pandas as pd
import numpy as np

filepath = "../results/BASE-S1D1O1-jun13"
label = filepath.split('/')[-1]

categories = ['ANC_REG', 'ANC_EXC', 'ANC_AMB', 'PRO_REG', 'PRO_EXC', 'PRO_AMB']

def calculate_stress(x):
   x = np.array(x)
   return np.sum(x*np.log2(x) + (1-x)*np.log2(1-x) - np.log2(0.5))

if __name__ == "__main__":
   # load data
   hl_df = pd.read_csv(f'{filepath}/warping-dilution-{label}-Hidden Layer.csv.gz',
                     converters={'activation': eval})

   print('Hidden Layer')
   for epoch in [350, 700]:
      print(f'--Epoch {epoch}')
      temp_df = hl_df[hl_df['epoch'] == epoch] # extract data of particular epoch
      temp_df["stress"] = temp_df['activation'].apply(calculate_stress)
      temp_df.to_csv(f'{filepath}/warping-dilution-{label}-Hidden Layer Stress Epoch {epoch}.csv.gz')

      for category in categories:
         print(f'   > {category}: {temp_df[temp_df["category"] == category]["stress"].mean()}')
      
      print(f'   > Base vocabulary: {temp_df[~temp_df["category"].isin(categories)]["stress"].mean()}')

   # load data
   ol_df = pd.read_csv(f'{filepath}/warping-dilution-{label}-Output Layer.csv.gz',
                     converters={'activation': eval})

   print('Output Layer')
   for epoch in [350, 700]:
      print(f'--Epoch {epoch}')
      temp_df = ol_df[ol_df['epoch'] == epoch] # extract data of particular epoch
      temp_df["onset_stress"] = temp_df['activation'].apply(lambda x: calculate_stress(x[:23]))
      temp_df["vowel_stress"] = temp_df['activation'].apply(lambda x: calculate_stress(x[23:37]))
      temp_df["coda_stress"] = temp_df['activation'].apply(lambda x: calculate_stress(x[37:]))
      temp_df.to_csv(f'{filepath}/warping-dilution-{label}-Output Layer Stress Epoch {epoch}.csv.gz')
      
      for category in categories:
         print(f'   > {category}, Onset: {temp_df[temp_df["category"] == category]["onset_stress"].mean()}')
         print(f'   > {category}, Vowel: {temp_df[temp_df["category"] == category]["vowel_stress"].mean()}')
         print(f'   > {category}, Coda: {temp_df[temp_df["category"] == category]["coda_stress"].mean()}')
      
      print(f'   > Base vocabulary, Onset: {temp_df[~temp_df["category"].isin(categories)]["onset_stress"].mean()}')
      print(f'   > Base vocabulary, Vowel: {temp_df[~temp_df["category"].isin(categories)]["vowel_stress"].mean()}')
      print(f'   > Base vocabulary, Codas: {temp_df[~temp_df["category"].isin(categories)]["coda_stress"].mean()}')