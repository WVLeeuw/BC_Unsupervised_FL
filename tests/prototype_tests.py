from blockchain import Blockchain
from block import Block
from device import Device, DevicesInNetwork
import KMeans
from utils import data_utils, stats_utils

from sklearn import cluster
from sklearn.datasets import load_breast_cancer, load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_genesis_block_with_params():
    bc = Blockchain()
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    vals = []
    for col in df.columns:
        vals.append([min(df[col]), max(df[col])])

    init_centroids = KMeans.randomly_init_centroid_range(vals, len(df.columns), 3)
    data = dict()
    data['centroids'] = init_centroids
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()
    block = Block(data=data, previous_hash=prev_block.get_hash())
    bc.mine(block)

    print(str(bc.get_chain_structure()[-1]))
    return bc


def test_block_retrieval_and_local_update():
    bc = test_genesis_block_with_params()
    recent_block = bc.get_most_recent_block()
    centroids = recent_block.get_data()['centroids']

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    model = cluster.KMeans(n_clusters=centroids.shape[0], init=centroids, n_init=1)
    model.fit(df)

    print(centroids)
    print(model.cluster_centers_)
