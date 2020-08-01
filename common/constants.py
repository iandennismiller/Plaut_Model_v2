"""
constants.py

=== SUMMARY ===
Description     : Load information from config file
Date Created    : July 19, 2020
Last Updated    : July 29, 2020

=== UPDATE NOTES ===
 > July 29, 2020
    - file documentation added
 > July 19, 2020
    - file created
"""


class VectorMapping:
    grapheme_onset = ['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G', 'F', 'V', 'J', 'Z', 'L', 'M', 'N', 'R', 'W',
                      'H', 'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH']
    grapheme_vowel = ['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 'EA', 'EE', 'EI', 'EU', 'EW', 'EY', 'IE',
                      'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY']
    grapheme_codas = ['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 'F', 'V', 'J', 'S', 'Z', 'P', 'T', 'K', 'Q',
                      'BB', 'CH', 'CK', 'DD', 'DG', 'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 'NG', 'NN', 'PH', 'PP', 'PS',
                      'RR', 'SH', 'SL', 'SS', 'TCH', 'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED']
    phoneme_onset = ['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D', 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h',
                     'l', 'r', 'w', 'y']
    phoneme_vowel = ['a', 'e', 'i', 'o', 'u', '@', '^', 'A', 'E', 'I', 'O', 'U', 'W', 'Y']
    phoneme_codas = ['r', 'l', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks', 'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S',
                     'Z', 'T', 'D', 'C', 'j']


class WordTypes:
    anchor_mapping = {'ANC_REG': 'Regular Anchors',
                      'ANC_EXC': 'Exception Anchors',
                      'ANC_AMB': 'Ambiguous Anchors'}
    probe_mapping = {'PRO_REG': 'Regular Probes',
                     'PRO_EXC': 'Exception Probes',
                     'PRO_AMB': 'Ambiguous Probes'}

    plaut_types = ['HEC', 'HRI', 'HFE', 'LEC', 'LFRI', 'LFE']
    anchor_types = anchor_mapping.keys()
    probe_types = probe_mapping.keys()
