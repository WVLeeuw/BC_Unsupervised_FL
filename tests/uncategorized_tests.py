from hashlib import sha256
from Crypto.PublicKey import RSA

from sklearn.datasets import load_breast_cancer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from utils import data_utils


def test_simple_hash():
    msg = 'hello world'.encode('utf-8')
    h = sha256(msg).digest()
    h_hex = sha256(msg).hexdigest()

    print(h == h_hex, type(h) == type(h_hex), h, h_hex)


def test_simple_encrypt():
    kp = RSA.generate(bits=1024)
    modulus = kp.n
    priv_key = kp.d
    pub_key = kp.e

    msg = 'hello world'.encode('utf-8')
    h = int.from_bytes(sha256(msg).digest(), byteorder='big')
    signature = pow(h, pub_key, modulus)
    print("Signature:", hex(signature))

    hash_from_signature = pow(signature, priv_key, modulus)
    print("Signature valid:", h == hash_from_signature)


def test_data_load_sklearn():
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    print(np.count_nonzero(breast_cancer.target))
    # print(df.columns)
    print(df.head())


def test_data_load_local():
    df = pd.read_csv("../data/breast_cancer_wisconsin.csv")
    # print(df.columns)
    print(df.head())


def test_data_load():
    train, test = data_utils.load(prop_train=.8, prop_test=.2)
    print(train.head())
    print(test.head())


def test_data_load_numpy():
    data, means = data_utils.create_dummy_data(dims=2, clients_per_cluster=1, samples_each=20, clusters=3)
    for c in data:
        plt.scatter(c[:, 0], c[:, 1])
    plt.show()
    print(means)


def test_data_load_make_blobs():
    X, y = data_utils.create_blobs(dims=2, verbose=True)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def test_make_blobs_divide():
    X, y = data_utils.create_blobs(dims=2, samples=300, verbose=True)
    samples_per_client = 100
    x1, x2, x3 = np.array_split(X, len(X)/samples_per_client)
    plt.scatter(x1[:, 0], x1[:, 1], color='blue')
    plt.scatter(x2[:, 0], x2[:, 1], color='red')
    plt.scatter(x3[:, 0], x3[:, 1], color='pink')
    plt.show()


def test_make_blobs_split():
    X_test, X_train, y_test, y_train = data_utils.create_blobs(dims=2, samples=300, split_train_test=True)
    print(X_train.shape)
    print(X_test.shape)


def test_make_blobs_divide_alternative():
    X, y = data_utils.create_blobs(dims=2, samples=300)
    num_devices = 3
    data_size_local = len(X) // num_devices

    datasets = []
    for i in range(num_devices):
        # print(i, i+1)
        # print(i*data_size_local, (i+1)*data_size_local)
        local_data = X[i*data_size_local:(i+1)*data_size_local]
        print(local_data.shape)
        datasets.append(local_data)

    # print(datasets)


def test_real_data_divide():
    train, test = data_utils.load(prop_train=.8, prop_test=.2)
    num_devices = 3

    train_size_local = len(train) // num_devices
    test_size_local = len(test) // num_devices

    dfs_train = []
    dfs_test = []
    for i in range(num_devices):
        local_data = train.iloc[i*train_size_local:(i+1)*train_size_local]
        local_test_data = test.iloc[i*test_size_local:(i+1)*test_size_local]
        print(local_data.shape, local_test_data.shape)
        dfs_train.append(local_data)
        dfs_test.append(local_test_data)

    for df in dfs_train:
        print(df.head())


def test_real_data_divide_lazy():
    train, test = data_utils.load(prop_train=.8, prop_test=.2)
    num_devices = 3

    dfs = np.array_split(train, num_devices)  # N.B. np.array_split works for ndarrays as well as for dataframes.
    for df in dfs:
        print(df.head())


# Test scipy's euclidean distance function, to be used to find the nearest global centroid from a local centroid.
def test_euclidean_distance():
    x = [0, 1]
    y = [1, 0]
    print(distance.euclidean(x, y))
