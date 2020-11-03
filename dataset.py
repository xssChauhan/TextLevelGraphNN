from typing import Tuple
import numpy as np

from torch.utils.data import Dataset
import torch

import utils

class R8Dataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        self.load_dataset()

    def load_dataset(self):

        self.data = utils.read_dataset(
            self.path, preprocessing=True
        )

        self.classes = np.unique(self.data.labels)

        self.classes_to_idx = {
            e:i
            for i,e in enumerate(self.classes)
        }

        self.vocab = utils.get_vocab(
            self.data.texts.tolist()
        )

        self.data['indexed_text'] = self.data.texts.map(
            lambda t: torch.tensor([self.vocab[e] for e in t])
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, x) -> Tuple[int, np.array]:
        '''
        Return the item at xth index. 
        Map the words to indices.
        '''

        row = self.data.iloc[x]
        text = row.indexed_text
        label = row.labels

        return self.classes_to_idx[label], text