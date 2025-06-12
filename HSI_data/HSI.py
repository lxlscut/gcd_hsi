from copy import deepcopy
import numpy as np
import torch
from HSI_data.data_utils import subsample_instances, MergedDataset
from HSI_data.datasets import Datasets


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.x = dataset.x[idxs]
        dataset.y = dataset.y[idxs]
        dataset.index = dataset.index[idxs]
        dataset.lo_index = dataset.lo_index[:,idxs]
        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1)):

    cls_idxs = [x for x, t in enumerate(dataset.y) if t in include_classes]

    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset



def get_hsi_datasets(whole_dataset, train_transform, test_transform,train_classes, prop_train_labels=0.8,seed=0):
    torch.random.manual_seed(seed)

    x = whole_dataset.x
    y = whole_dataset.y
    lo_index = whole_dataset.lo_index
    index = whole_dataset.index

    # Split the indices into train and test sets
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(index, test_size=0.2, random_state=42)

    train_dataset = Datasets(x = x[train_indices,:],y = y[train_indices],lo_index = lo_index[:,train_indices],index = index[train_indices],transform = train_transform)
    test_dataset = Datasets(x = x[test_indices,:],y = y[test_indices],lo_index = lo_index[:,test_indices],index = index[test_indices],transform = test_transform)
    train_dataset.index = np.arange(train_dataset.x.shape[0])

    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)

    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    unlabelled_indices = set(train_dataset.index) - set(train_dataset_labelled.index)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

    train_dataset_m = MergedDataset(labelled_dataset=deepcopy(train_dataset_labelled),
                                  unlabelled_dataset=deepcopy(train_dataset_unlabelled))

    test_dataset = test_dataset
    unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset_m, test_dataset, unlabelled_train_examples_test, train_dataset
