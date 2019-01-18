import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236605.dataloader_utils as dataloader_utils
from . import dataloaders
from . import datasets


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)


        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        
        _, topk_indices = torch.topk(dist_matrix, self.k, dim=0, largest=False)
        # print(topk_indices)

        for i in range(n_test):

            classes = {}
            for j in range(self.k):
                jth_class = self.y_train[topk_indices[j][i].item()].item()
                if jth_class not in classes:
                    classes[jth_class] = 0
                classes[jth_class] += 1
                
            most_frequent, top_freq = None, 0
            for c, freq in classes.items():
                if most_frequent is None or top_freq < freq:
                    most_frequent, top_freq = c, freq
            y_pred[i] = most_frequent


        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """
        train_product = torch.sum(self.x_train ** 2, dim=1)
        test_product = torch.sum(x_test ** 2, dim=1)
        
        pairwise_product = torch.mm(self.x_train, torch.transpose(x_test, 0, 1))
        
        train_col = train_product.view(pairwise_product.shape[0], 1)
        test_row = test_product.view(1, pairwise_product.shape[1])

        dists = train_col + test_row - 2 * pairwise_product


        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    # print(torch.sum(torch.eq(y, y_pred)).item())
    # print(y.shape[0])
    return torch.sum(torch.eq(y, y_pred)).item() / float(y.shape[0])
    # ========================

    return accuracy

class WithoutSubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """
    def __init__(self, source_dataset: Dataset, removed_subset_len, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """

        if offset + removed_subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.removed_subset_len = removed_subset_len
        self.offset = offset

    def __getitem__(self, index):
        if index < 0 or index >= len(self.source_dataset) - self.removed_subset_len:
            raise IndexError("Index out of bounds")
        if index < self.offset:
            return self.source_dataset[index]
        else:
            return self.source_dataset[index + self.removed_subset_len]

    def __len__(self):
        # ====== YOUR CODE: ======
        return len(self.source_dataset) - self.removed_subset_len
        # ========================


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        accuracies_for_k = []
        for j in range(num_folds):
            removed_offset = len(ds_train) * j // num_folds
            removed_len = min(len(ds_train) // num_folds,
                                 len(ds_train) - removed_offset)
            ds_actual_train = WithoutSubsetDataset(
                ds_train, removed_len, offset=removed_offset)
            ds_actual_valid = datasets.SubsetDataset(
                ds_train, removed_len, offset=removed_offset)

            knn_classifier = KNNClassifier(k=k)

            batch_size = 1024
            knn_classifier.train(torch.utils.data.DataLoader(ds_actual_train,
                                                             batch_size))

            x_valid, y_valid = dataloader_utils.flatten(
                torch.utils.data.DataLoader(ds_actual_valid, batch_size))

            y_pred = knn_classifier.predict(x_valid)

            # Calculate accuracy
            valid_accuracy = accuracy(y_valid, y_pred)
            accuracies_for_k.append(valid_accuracy)
            
        accuracies.append(accuracies_for_k)
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
