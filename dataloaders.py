import math

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from torch.utils.data import Dataset

from hw1.datasets import SubsetDataset


def create_train_validation_loaders(dataset: Dataset, validation_ratio,
                                    batch_size=100, num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    dataset_size = len(dataset)
    permutation = torch.randperm(dataset_size)
    valid_size = int(dataset_size * validation_ratio)
    train_size = dataset_size - valid_size
    
    valid_sampler = torch.utils.data.SubsetRandomSampler(permutation[:valid_size])
    train_sampler = torch.utils.data.SubsetRandomSampler(permutation[valid_size:])
    
    dl_valid = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=num_workers,
        sampler=valid_sampler
    )
    dl_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler
    )
    return dl_train, dl_valid

