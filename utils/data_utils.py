import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

project_dir = os.path.dirname(os.getcwd())

# This file describes utility functions for loading and creating (dummy) datasets.


def load(subsample_train_frac=None, num_train=None, num_test=None, seed=None, verbose=None):
    np.random.seed(seed)
    # This is where we locate the .csv files and read them as pandas dataframes.
    pass


def create_dummy_data(dims=1, clients_per_cluster=10, samples_each=10, clusters=10, scale=.5, verbose=False):
    num_clients = clients_per_cluster * clusters
    # create gaussian data set, per client one mean
    means = np.arange(1, clusters+1)
    means = np.tile(A=means, reps=clients_per_cluster)
    noise = np.random.normal(loc=0.0, scale=scale, size=(num_clients, samples_each, dims))
    data = np.expand_dims(np.expand_dims(means, axis=1), axis=2) + noise
    if verbose:
        print(means)
        print(noise)
        print("dummy data shape: ", data.shape)
    data = [data[i] for i in range(num_clients)]
    return data, means


def load_federated_dummy(seed=None, verbose=False, clients_per_cluster=10, clusters=10):
    np.random.seed(seed)
    x = {}
    ids = {}
    data, means = create_dummy_data(clients_per_cluster=2*clients_per_cluster, clusters=clusters, verbose=verbose)
    mid = clients_per_cluster * clusters
    x["train"], ids["train"] = data[:mid], means[:mid]
    x["test"], ids["test"] = data[mid:], means[mid:]
    print(len(x['train']), x['train'][0].shape)


def load_federated(limit_csv=None, verbose=False, seed=None, dummy=False, clusters=None):
    if dummy:
        return load_federated_dummy(seed=seed, verbose=verbose, clusters=clusters)
    else:
        return load_federated_real(limit_csv, verbose, seed)


def load_federated_real(limit_csv=None, verbose=False, seed=None):
    pass


def test():
    pass


if __name__ == "__main__":
    test()
