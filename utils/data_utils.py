import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_blobs
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
    eligible_datasets = ['iris', 'breast cancer', 'wine', 'heart disease', 'forest types', 'blobs', 'dummy']
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
            return split_data(df.to_numpy(), labels, num_devices, split_train_test, prop_test, is_iid)

        if dataset == 'breast cancer':
            breast_cancer = load_breast_cancer()
            df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
            labels = breast_cancer.target
            return split_data(df.to_numpy(), labels, num_devices, split_train_test, prop_test, is_iid)

        if dataset == 'wine':
            wine = load_wine()
            df = pd.DataFrame(wine.data, columns=wine.feature_names)
            labels = wine.target
            return split_data(df.to_numpy(), labels, num_devices, split_train_test, prop_test, is_iid)

        if dataset == 'heart disease':
            df = pd.read_csv("data/heart_disease_cleveland.csv")  # not sure how to retrieve just labels here...
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
            cur_file = os.path.abspath(os.path.dirname(__file__))
            covtype_file = os.path.join(cur_file, 'data/forest_covertypes.csv')
            df = pd.read_csv(covtype_file)
            labels = df['Cover_Type']
            df = df.drop('Cover_Type', axis=1)
            print(df.shape)
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
                return np.array_split(df.to_numpy(), num_devices), np.array_split(labels.to_numpy(), num_devices)

        if dataset == 'dummy':
            clients_per_cluster = num_devices//clusters
            data, _ = create_dummy_data(dims=dims, clients_per_cluster=clients_per_cluster,
                                        samples_each=samples//num_devices, clusters=clusters, verbose=True)
            print(len(data))
            return data, _

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
            exp_train_per_device = round(len(df_train)/num_devices)
            exp_test_per_device = round(len(df_test)/num_devices)
            range_sizes_train = range(exp_train_per_device - exp_train_per_device//4,
                                      exp_train_per_device + exp_train_per_device//4)
            range_sizes_test = range(exp_test_per_device - exp_test_per_device//4,
                                     exp_test_per_device + exp_test_per_device//4)
            rand_sizes_train = [random.choice(range_sizes_train)]
            rand_sizes_test = [random.choice(range_sizes_test)]
            for i in range(1, num_devices):
                rand_sizes_train.append(random.choice(range_sizes_train) + rand_sizes_train[i - 1])
                rand_sizes_test.append(random.choice(range_sizes_test) + rand_sizes_test[i - 1])
            return np.split(df_test, rand_sizes_test), np.split(df_train, rand_sizes_train), \
                np.split(labels[:num_test][order_test], rand_sizes_test), \
                np.split(labels[num_test:][order_train], rand_sizes_train)  # test, train, y_test, y_train
        else:
            order = np.argsort(labels)
            exp_records_per_device = round(len(df)/num_devices)
            range_sizes = range(exp_records_per_device - exp_records_per_device//4,
                                exp_records_per_device + exp_records_per_device//4)
            rand_sizes = [random.choice(range_sizes)]
            for i in range(1, num_devices):
                rand_sizes.append(random.choice(range_sizes) + rand_sizes[i-1])
            return np.split(df[order], rand_sizes), np.split(labels[order], rand_sizes)  # X, y
    elif is_iid:
        if split_train_test:
            if prop_test is not None:
                prop = prop_test
            else:
                prop = .2
            num_test = int(prop * len(df))
            exp_train_per_device = round((len(df)-num_test)/num_devices)
            exp_test_per_device = round(num_test/num_devices)
            range_sizes_train = range(exp_train_per_device - exp_train_per_device//4,
                                      exp_train_per_device + exp_train_per_device//4)
            range_sizes_test = range(exp_test_per_device - exp_test_per_device//4,
                                     exp_test_per_device + exp_test_per_device//4)
            rand_sizes_train = [random.choice(range_sizes_train)]
            rand_sizes_test = [random.choice(range_sizes_test)]
            for i in range(1, num_devices):
                rand_sizes_train.append(random.choice(range_sizes_train) + rand_sizes_train[i-1])
                rand_sizes_test.append(random.choice(range_sizes_test) + rand_sizes_test[i-1])
            return np.split(df[:num_test], rand_sizes_test[:-1]), np.split(df[num_test:], rand_sizes_train[:-1]), \
                np.split(labels[:num_test], rand_sizes_test), np.split(labels[num_test:], rand_sizes_train)
            # test, train, y_test, y_train
        else:
            exp_records_per_device = round(len(df) / num_devices)
            range_sizes = range(exp_records_per_device - exp_records_per_device//4,
                                exp_records_per_device + exp_records_per_device//4)
            rand_sizes = [random.choice(range_sizes)]
            for i in range(1, num_devices):
                rand_sizes.append(random.choice(range_sizes) + rand_sizes[i-1])
            return np.split(df, rand_sizes), np.split(labels, rand_sizes)  # X, y
    else:  # this case should never be reached, but we include it anyway
        exp_records_per_device = round(len(df) / num_devices)
        range_sizes = range(exp_records_per_device - exp_records_per_device//4,
                            exp_records_per_device + exp_records_per_device//4)
        rand_sizes = []
        for i in range(num_devices):
            rand_sizes.append(random.choice(range_sizes))
        return np.split(df, rand_sizes), np.split(labels, rand_sizes)  # X, y


# Function that returns the bounds (min, max) for each dimension in dataset df. Expects df to be a np.array.
# Should be a matrix describing n records having i features (dimensions).
def obtain_bounds(df):
    min_vals = df.min(axis=0)
    max_vals = df.max(axis=0)
    if isinstance(min_vals, pd.Series):
        min_vals = min_vals.to_numpy()
    if isinstance(max_vals, pd.Series):
        max_vals = max_vals.to_numpy()
    return min_vals.flatten(), max_vals.flatten()


def obtain_bounds_multiple(dfs):
    min_vals, max_vals = obtain_bounds(np.asarray(dfs[0]))
    for df in dfs[1:]:
        min_vals_df, max_vals_df = obtain_bounds(df)
        for dim in range(len(min_vals)):
            if min_vals_df[dim] < min_vals[dim]:
                min_vals[dim] = min_vals_df[dim]
            if max_vals_df[dim] > max_vals[dim]:
                max_vals[dim] = max_vals_df[dim]
    if isinstance(min_vals, pd.Series):
        min_vals = min_vals.to_numpy()
    if isinstance(max_vals, pd.Series):
        max_vals = max_vals.to_numpy()
    return min_vals, max_vals


def create_dummy_data(dims=1, clients_per_cluster=10, samples_each=10, clusters=10, scale=.5, verbose=False):
    np.random.seed(42)  # 42, 420, 4200, 42000, 420000 = 5 runs.
    num_clients = clients_per_cluster * clusters
    print(f"Number of clients set to: {num_clients}.")
    # create gaussian data set, per client one mean
    means = np.arange(1, clusters + 1)
    means = np.tile(A=means, reps=clients_per_cluster)
    noise = np.random.normal(loc=0.0, scale=scale, size=(num_clients, samples_each, dims))
    data = np.expand_dims(np.expand_dims(means, axis=1), axis=2) + noise
    if verbose:
        # print(means)
        # print(noise)
        print("dummy data shape: ", data.shape)
    data = [data[i] for i in range(num_clients)]
    return data, means


# Function to create blobs that are very clearly structured as clusters.
# Can already obtain train and test sets from this within this function.
def create_blobs(dims=2, samples=100, clusters=3, split_train_test=False, prop_test=None, is_iid=True, num_devices=3,
                 verbose=False):
    X, y = make_blobs(n_samples=samples, centers=clusters, n_features=dims, random_state=42)

    if verbose:
        print(X.shape)
        print(len(y), y)

    return split_data(X, y, num_devices, split_train_test, prop_test, is_iid)
