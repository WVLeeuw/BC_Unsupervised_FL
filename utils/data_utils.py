import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, make_blobs
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


def load_data(dataset=None, is_iid=True, num_devices=3, split_train_test=False, prop_test=None, dims=2, samples=100,
              clusters=3, verbose=False):
    eligible_datasets = ['iris', 'breast cancer', 'heart disease', 'forest types', 'blobs']
    if dataset not in eligible_datasets:
        print("No dataset was requested or the requested dataset could not be retrieved.")
        print("Defaulting to generating blobs...")
        return create_blobs(dims=dims, samples=samples, clusters=clusters, split_train_test=split_train_test,
                            prop_test=prop_test, is_iid=is_iid, num_devices=num_devices, verbose=verbose)
    else:
        if dataset == 'iris':
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            labels = iris.target
            return split_data(df, labels, num_devices, split_train_test, prop_test, is_iid)

        if dataset == 'breast cancer':
            breast_cancer = load_breast_cancer()
            df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
            labels = breast_cancer.target
            return split_data(df, labels, num_devices, split_train_test, prop_test, is_iid)

        if dataset == 'heart disease':
            df = pd.read_csv("../data/heart_disease_cleveland.csv")  # not sure how to retrieve just labels here...
            # this implementation does not use split_data for now, because it would require us to extract labels.
            if split_train_test and prop_test is not None:
                df_train, df_test = train_test_split(df, test_size=prop_test)
                if not is_iid:
                    return NotImplementedError
                else:
                    return np.array_split(df_train, num_devices), np.array_split(df_test, num_devices)
            elif split_train_test:
                df_train, df_test = train_test_split(df, test_size=.2)
                if not is_iid:
                    return NotImplementedError
                else:
                    return np.array_split(df_train, num_devices), np.array_split(df_test, num_devices)
            else:
                return np.array_split(df, num_devices)

        if dataset == 'forest types':
            df = pd.read_csv("../data/forest_covertypes.csv")  # not sure how to retrieve just labels here...
            # this implementation does not use split_data for now, because it would require us to extract labels.
            if split_train_test and prop_test is not None:
                df_train, df_test = train_test_split(df, test_size=prop_test)
                if not is_iid:
                    return NotImplementedError
                else:
                    return np.array_split(df_train, num_devices), np.array_split(df_test, num_devices)
            elif split_train_test:
                df_train, df_test = train_test_split(df, test_size=.2)
                if not is_iid:
                    return NotImplementedError
                else:
                    return np.array_split(df_train, num_devices), np.array_split(df_test, num_devices)
            else:
                if is_iid:
                    return NotImplementedError
                else:
                    return np.array_split(df, num_devices)

        else:  # default case
            print("Generating blobs...")
            create_blobs(dims=dims, samples=samples, clusters=clusters, split_train_test=split_train_test,
                         prop_test=prop_test, is_iid=is_iid, num_devices=num_devices, verbose=verbose)


def split_data(df, labels, num_devices, split_train_test=False, prop_test=None, is_iid=True):
    if not is_iid:
        if split_train_test:
            if prop_test is not None:
                prop = prop_test
            else:
                prop = .2
            num_test = int(prop * len(df))
            order_train = np.argsort(labels[num_test:])
            order_test = np.argsort(labels[:num_test])
            df_train = df[order_train]
            df_test = df[order_test]
            return np.array_split(df_test, num_devices), np.array_split(df_train, num_devices), \
                np.array_split(labels[:num_test][order_test], num_devices), \
                np.array_split(labels[num_test:][order_train], num_devices)  # test, train, y_test, y_train
        else:
            order = np.argsort(labels)
            return np.array_split(df[order], num_devices), np.array_split(labels[order], num_devices)  # X, y
    elif is_iid:
        if split_train_test:
            if prop_test is not None:
                prop = prop_test
            else:
                prop = .2
            num_test = int(prop * len(df))
            return np.array_split(df[:num_test], num_devices), np.array_split(df[num_test:], num_devices), \
                np.array_split(labels[:num_test], num_devices), np.array_split(labels[num_test:], num_devices)
            # test, train, y_test, y_train
        else:
            return np.array_split(df, num_devices), np.array_split(labels, num_devices)
    else:  # this case should never be reached, but we include it anyway
        return np.array_split(df, num_devices), np.array_split(labels, num_devices)


def create_dummy_data(dims=1, clients_per_cluster=10, samples_each=10, clusters=10, scale=.5, verbose=False):
    num_clients = clients_per_cluster * clusters
    # create gaussian data set, per client one mean
    means = np.arange(1, clusters + 1)
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
def create_blobs(dims=2, samples=100, clusters=3, split_train_test=False, prop_test=None, is_iid=True, num_devices=3,
                 verbose=False):
    X, y = make_blobs(n_samples=samples, centers=clusters, n_features=dims)

    if verbose:
        print(X.shape)
        print(len(y), y)

    return split_data(X, y, num_devices, split_train_test, prop_test, is_iid)
