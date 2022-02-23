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


def test_make_blobs_divide():
    X, y = data_utils.create_blobs(dims=2, samples=300, verbose=True)
    colors = ['blue', 'red', 'pink']
    for i in range(len(X)):
        plt.scatter(X[i][:, 0], X[i][:, 1], color=colors[i])
    plt.show()


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


def test_non_iid_synth_data():
    dfs, labels = data_utils.load_data(is_iid=False)
    print(dfs[0][:5], labels[0][:5])
    print(len(dfs))

    colors = ['blue', 'red', 'pink']
    for i in range(len(dfs)):
        plt.scatter(dfs[i][:, 0], dfs[i][:, 1], color=colors[i])
    plt.show()


def test_load_data_real():
    dfs, labels = data_utils.load_data(dataset='iris')
    print(dfs[0][:5], labels[0][:5])

    dfs, labels = data_utils.load_data(dataset='breast cancer')
    print(dfs[0][:5], labels[0][:5])


def test_obtain_bounds():
    dfs, labels = data_utils.load_data()
    print(dfs[0])
    min_vals, max_vals = data_utils.obtain_bounds(np.asarray(dfs[0]))
    print(min_vals, max_vals)
    for dim in range(len(min_vals)):
        print(min_vals[dim])
        print(max_vals[dim])

    # plt.scatter(dfs[0][:, 0], dfs[0][:, 1], color='green')
    # plt.show()


def test_obtain_bounds_multiple():
    # dfs, labels = data_utils.load_data()
    dfs = [[[1, 2], [3, 3]], [[0, 0], [2, -2]], [[-4, 0], [2, -1]]]
    print(data_utils.obtain_bounds_multiple(np.asarray(dfs)))
    min_vals, max_vals = data_utils.obtain_bounds_multiple(np.asarray(dfs))
    bounds = []
    for i in range(len(min_vals)):
        bounds.append([min_vals[i], max_vals[i]])
    print(bounds)

