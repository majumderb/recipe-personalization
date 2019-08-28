'''
-*- coding: utf-8 -*-

Sampling and dataset loading

@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}

Copyright Shuyang Li & Bodhisattwa Majumder
License: GNU GPLv3
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import os

# https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
def _flatten(list_of_lists):
    for x in list_of_lists:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in _flatten(x):
                yield y
        else:
            yield x

def collate(samples):
    return [torch.LongTensor(c) for c in zip(*samples)]

class DataFrameDataset(torch.utils.data.Dataset):
    """
    Dataset for data from arbitrary pandas DataFrames
    """
    def __init__(self, df, col_order=None):
        super().__init__()

        self.df = df
        self.col_order = col_order or list(self.df.columns)
        self.flattened_col_order = list(_flatten(self.col_order))
        self.df_view = self.df[self.flattened_col_order]

        print('Loaded dataset with {:,} rows'.format(
            len(self.df),
        ))

    def set_col_order(self, col_order):
        '''
        Mutate column order for this same dataset
        
        Arguments:
            col_order {list} -- List of relevant columns in order
        
        Returns:
            DataFrameDataset -- Mutated dataset
        '''
        self.col_order = col_order
        self.flattened_col_order = list(_flatten(self.col_order))
        self.df_view = self.df[self.flattened_col_order]
        return self

    def __len__(self):
        '''
        Returns:
            int -- Number of examples in this DataSet
        '''
        return len(self.df)

    def __getitem__(self, i):
        '''
        Gets a single example
        '''
        # Get single row
        row = self.df_view.iloc[i]

        # This will most likely be a list of u, i, j, [i stuff], [j stuff]
        return_items = [row[cols] for cols in self.col_order]
        return return_items
    
    def get_tensor_batch(self, indices=None):
        if indices is None:
            batch_df = self.df_view
        else:
            batch_df = self.df_view.iloc[indices]
        return [torch.LongTensor(batch_df[cols].values) for cols in self.col_order]

class BatchSampler(object):
    """
    Simple sampler for a DataFrameDataset
    """
    def __init__(self, dataset, batch_size=2048, n_batches=None, random=False):
        """
        Arguments:
            dataset {DataFrameDataset} -- DataFrame-based dataset

        Keyword Arguments:
            batch_size {int} -- Number of examples per batch. If specified, `n_batches` argument is ignored. (default: {2048})
            n_batches {int} -- Number of batches per epoch (default: {None})
            shuffled {bool} -- Whether to randomly shuffle batches. (default: {False})
        """
        self.dataset = dataset
        self.epoch_size = len(dataset)
        self.indices = np.arange(self.epoch_size)
        self.random = random
        self.renew_indices()

        # Get batch sizes
        if batch_size:
            self.batch_size = batch_size
            self.n_batches = int(np.ceil(self.epoch_size / self.batch_size))
        elif n_batches:
            self.n_batches = n_batches
            self.batch_size = int(np.ceil(self.epoch_size / self.n_batches))
        else:
            raise Exception(
                'Must specify either a batch size (`batch_size`) or number of batches (`n_batches`)!'
            )

        print('Every epoch, we have {:,} {:,}-sized batches for a total of {:,} instances'.format(
            self.n_batches, self.batch_size, self.epoch_size
        ))

    def renew_indices(self):
        if self.random:
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(self.epoch_size)

    def epoch_batches(self):
        for batch_start in range(self.n_batches):
            batch_ix = self.indices[
                batch_start * self.batch_size : (batch_start  + 1) * self.batch_size
            ]
            yield self.dataset.get_tensor_batch(indices=batch_ix)
        
        self.renew_indices()
    
    def __len__(self):
        return self.n_batches
