from common.helpers import get_graphemes, get_phonemes

import pandas as pd
from tqdm import tqdm


class LensRepresentation:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, na_filter=False, dtype={'freq': float, 'log_freq': float})

    def create_lens_file(self):
        text = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            for col in self.df.columns:
                text.append(f"name: {row[col]}")
                text.append(f"I: {' '.join([str(x) for x in get_graphemes(row[col])])}")
                text.append(';')
        # for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
        #     text.append(f"name: {row['orth']}")
        #     text.append(f"freq: {row['log_freq']}")
        #     text.append(f"I: {' '.join([str(x) for x in get_graphemes(row['orth'])])}")
        #     text.append(f"T: {' '.join([str(x) for x in get_phonemes(row['phon'])])}")
        #     text.append(';')

        with open('lens.txt', 'w') as f:
            f.write('\n'.join(text))
