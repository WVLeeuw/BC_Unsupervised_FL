import KMeans

from sklearn import cluster
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_centroid_init():
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    values = [[1, 10], [0, 4]]
    centroid = KMeans.randomly_init_centroid_range(values, 2)  # Try initializing a single centroid
    centroids = KMeans.randomly_init_centroid_range(values, 2, 3)  # Idem, but multiple
    print(centroid)
    print(centroids)


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
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    kmeans.predict(test)


def test_sklearn_kmeans_real():
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

    # Train-test split
    train, test = train_test_split(df, test_size=.2)

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(train)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

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

    train, test = train_test_split(df, test_size=.2)

    kmeans_length = cluster.KMeans(n_clusters=3)
    kmeans_length.fit(train[['sepal length (cm)', 'petal length (cm)']])
    label_length = kmeans_length.fit_predict(test[['sepal length (cm)', 'petal length (cm)']])

    test_length = test[['sepal length (cm)', 'petal length (cm)']]
    centroids = kmeans_length.cluster_centers_
    u_labels_length = np.unique(label_length)
    inertia = kmeans_length.inertia_
    print(inertia/len(test))

    for i in u_labels_length:
        plt.scatter(test_length[label_length == i].iloc[:, 0], test_length[label_length == i].iloc[:, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.xlabel(test_length.columns[0])
    plt.ylabel(test_length.columns[1])
    plt.legend()
    plt.show()


def test_kmeans_pca():
    iris = load_iris().data
    pca = PCA(2)
    df = pca.fit_transform(iris)

    kmeans = cluster.KMeans(n_clusters=3)
    label = kmeans.fit_predict(df)
    u_labels = np.unique(label)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    print(inertia/len(df))

    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.show()
