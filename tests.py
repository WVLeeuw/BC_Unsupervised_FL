import sklearn.cluster

from blockchain import Blockchain
from block import Block
from device import Device, DevicesInNetwork
import KMeans
from utils import data_utils, stats_utils

from hashlib import sha256
from Crypto.PublicKey import RSA
from sklearn import cluster
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_bc_genesis():
    bc = Blockchain()
    bc.create_genesis_block()
    print(str(bc.get_chain_structure()[0]))


def test_block_propagation():
    bc = Blockchain()
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()
    # Test whether we acquire the same hash if we rehash the most recent block.
    print(sha256(str(bc.get_most_recent_block()).encode('utf-8')).hexdigest() == prev_block.get_hash().hexdigest())

    # Test whether we can add a new block using that hash.
    block = Block(data='test_block', previous_hash=prev_block.get_hash())
    bc.mine(block)
    added_properly = bc.get_chain_length()
    if added_properly:
        print(str(bc.get_most_recent_block()))


def test_nonce_increment():
    bc = Blockchain(difficulty=1)
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()

    block1 = Block(data='first_test_block', previous_hash=prev_block.get_hash())
    bc.mine(block1)

    recent_block = bc.get_most_recent_block()
    block2 = Block(data='second_test_block', previous_hash=recent_block.get_hash())
    bc.mine(block2)

    print(str(bc.get_chain_structure()[-2]), str(bc.get_chain_structure()[-1]))


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


def test_kmeans_dummy():
    kmeans = KMeans.init_kmeans_python(n_clusters=3)
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    centroids, labels = kmeans.fit(X=x)

    print(x.shape)
    print(kmeans.labels_)
    print(kmeans.predict(np.asarray([[0, 0], [12, 3]])))
    print(kmeans.cluster_centers_)


def test_sklearn_kmeans_dummy():
    n_clusters = 3
    seed = np.random.seed()
    init_centroids = 'random'

    # Initialize KMeans with scikit-learn
    model = cluster.KMeans(n_clusters=n_clusters, random_state=seed, init=init_centroids, max_iter=100, tol=.001,
                           n_init=1, precompute_distances=True, algorithm='full')
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    model.fit(X)

    print(X.shape)
    print(model.labels_)
    print(model.predict([[0, 0], [12, 3]]))
    print(model.cluster_centers_)


def test_data_load_sklearn():
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    print(np.count_nonzero(breast_cancer.target))
    # print(df.columns)
    print(df.head())


def test_data_load_local():
    df = pd.read_csv("data/breast_cancer_wisconsin.csv")
    # print(df.columns)
    print(df.head())


# For these tests, different implementations of k-means are used.
# Moreover, it does not matter whether we retrieve the datasets from sklearn.datasets or from local storage.
# However, using sklearn.datasets makes these tests executable remotely.
def test_kmeans_real():
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    labels = pd.Series(breast_cancer.target)

    # Train-test split
    train, test = train_test_split(df, test_size=.2)

    kmeans = KMeans.init_kmeans(n_clusters=3)
    kmeans.fit(train)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    kmeans.predict(test)


def test_sklearn_kmeans_real():
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    labels = pd.Series(breast_cancer.target)

    # Train-test split
    train, test = train_test_split(df, test_size=.2)

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(train)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    kmeans.predict(test)


def test_elbow_method():
    ks = range(1, 10)
    inertias = []
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = pd.Series(iris.target)

    for k in ks:
        model = cluster.KMeans(n_clusters=k)
        model.fit(df)
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


def test_kmeans_simple_vis():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = pd.Series(iris.target)

    train, test = train_test_split(df, test_size=.2)

    # kmeans = cluster.KMeans(n_clusters=3)
    # kmeans.fit(train)

    kmeans_length = cluster.KMeans(n_clusters=3)
    kmeans_length.fit(train[['sepal length (cm)', 'petal length (cm)']])
    label_length = kmeans_length.fit_predict(test[['sepal length (cm)', 'petal length (cm)']])

    test_length = test[['sepal length (cm)', 'petal length (cm)']]
    centroids = kmeans_length.cluster_centers_
    u_labels_length = np.unique(label_length)

    for i in u_labels_length:
        plt.scatter(test_length[label_length == i].iloc[:, 0], test_length[label_length == i].iloc[:, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.xlabel(test_length.columns[0])
    plt.ylabel(test_length.columns[1])
    plt.legend()
    plt.show()
