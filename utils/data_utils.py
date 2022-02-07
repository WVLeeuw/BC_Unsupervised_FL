import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split

project_dir = os.path.dirname(os.getcwd())

# This file describes utility functions for loading and creating (dummy) datasets.


def load(subsample_train_frac=None, prop_train=None, prop_test=None, is_iid=True, verbose=None):
    # This is where we locate the .csv files and read them as pandas dataframes.
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_train, df_test = train_test_split(df, test_size=prop_test)
    # print(df.columns)
    if verbose:
        print(df.head())
    return df_train, df_test


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


# Function to create blobs that are very clearly structured as clusters.
# Can already obtain train and test sets from this within this function.
def create_blobs(dims=2, samples=100, clusters=3, split_train_test=False, verbose=False):
    X, y = make_blobs(n_samples=samples, centers=clusters, n_features=dims)

    if verbose:
        print(X.shape)
        print(len(y), y)

    if split_train_test:
        prop = .2
        num_test = int(prop*len(X))
        return X[:num_test], X[num_test:], y[:num_test], y[num_test:]  # test, train, y_test, y_train

    return X, y


def load_federated_dummy(seed=None, verbose=False, clients_per_cluster=10, clusters=10):
    np.random.seed(seed)
    x = {}
    ids = {}
    data, means = create_dummy_data(clients_per_cluster=2*clients_per_cluster, clusters=clusters, verbose=verbose)
    mid = clients_per_cluster * clusters
    x["train"], ids["train"] = data[:mid], means[:mid]
    x["test"], ids["test"] = data[mid:], means[mid:]
    print(len(x['train']), x['train'][0].shape)
    return x, ids


def load_federated(limit_csv=None, verbose=False, seed=None, dummy=False, clusters=None):
    if dummy:
        return load_federated_dummy(seed=seed, verbose=verbose, clusters=clusters)
    else:
        raise NotImplementedError
