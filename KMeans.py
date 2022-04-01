import numpy as np
import random


def randomly_init_centroid(min_value, max_value, n_dims, repeats=1):
    if repeats == 1:
        return min_value + (max_value - min_value) * np.random.rand(n_dims)
    else:
        return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)


def randomly_init_centroid_range(values, n_dims, repeats=1):
    assert len(values) == n_dims, "Must supply a range of values that has equal length as the number of features in " \
                                  "the dataset. "
    if repeats == 1:
        centroid = []
        for value in values:
            centroid.append(value[0] + (value[1] - value[0]) * np.random.rand())
        return np.asarray(centroid)
    else:
        centroids = []
        for i in range(repeats):
            centroid = []
            for value in values:
                centroid.append(value[0] + (value[1] - value[0]) * np.random.rand())
            centroids.append(centroid)
        return np.asarray(centroids)


class KMeans:
    def __init__(self, n_clusters, init_centroids='random', max_iter=100, tol=0.0001, distance_metric='euclidean',
                 seed=None, reassign_min=None, reassign_after=None, n_dims=None, verbose=False):
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_clusters = n_clusters
        self.seed = seed
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        if distance_metric != 'euclidean':
            raise NotImplementedError
        self.n_dims = n_dims
        self.reassign_min = reassign_min
        self.reassign_after = reassign_after
        self.verbose = verbose

    def do_init_centroids(self):
        if isinstance(self.init_centroids, str):
            if self.init_centroids == 'random':
                centroids = randomly_init_centroid(0, self.n_clusters + 1, self.n_dims, self.n_clusters)
            else:
                raise NotImplementedError
        # elif self.init_centroids.shape == (self.n_clusters, self.n_dims):
        #     centroids = self.init_centroids
        else:
            raise NotImplementedError
        return centroids

    def fit(self, X, record_at=None):
        x = X
        self.n_dims = x.shape[1]
        centroids = self.do_init_centroids()
        means_record = []
        stds_record = []
        to_reassign = np.zeros(self.n_clusters)
        for iteration in range(0, self.max_iter):
            # compute distances
            sq_dist = np.zeros((x.shape[0], self.n_clusters))  # not sure what this does
            for i in range(self.n_clusters):
                sq_dist[:, i] = np.sum(np.square(x - centroids[i, :]), axis=1)  # sum the square distances(?)

            labels = np.argmin(sq_dist, axis=1)
            # update centroids
            centroid_updates = np.zeros((self.n_clusters, self.n_dims))

            for i in range(self.n_clusters):
                mask = np.equal(labels, i)
                size = np.sum(mask)
                if size > 0:
                    update = np.sum(x[mask] - centroids[i], axis=0)
                    centroid_updates[i, :] = update / size
                if self.reassign_min is not None:
                    if size < x.shape[0] * self.reassign_min:
                        to_reassign[i] += 1
                    else:
                        to_reassign[i] = 0

            centroids = centroids + centroid_updates
            changed = np.any(np.absolute(centroid_updates) > self.tol)

            if self.reassign_after is not None:
                for i, num_no_change in enumerate(to_reassign):
                    if num_no_change >= self.reassign_after:
                        centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.n_dims, 1)
                        to_reassign[i] = 0
                        changed = True

            if record_at is not None and iteration in record_at:
                means, stds = record_state(centroids, x)
                means_record.append(means)
                stds_record.append(stds)
            # if not changed:
            #     break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        return centroids, labels

    def predict(self, x):
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.cluster_centers_[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels


def init_kmeans_python(n_clusters, init_centroids='random', batch_size=None, seed=None, iterations=100, verbose=False):
    if batch_size is not None:
        raise NotImplementedError
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            init_centroids=init_centroids,
            seed=seed,
            max_iter=iterations,
            verbose=verbose
        )
    return kmeans


def init_kmeans(**kwargs):
    return init_kmeans_python(**kwargs)


def record_state(centroids, x):
    assert centroids.shape[1] == 1
    differences = np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)
    labels = np.argmin(sq_dist, axis=1)
    stds = np.zeros(centroids.shape[0])
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        counts = np.sum(mask)
        if counts > 0:
            stds[i] = np.std(x[mask])
    return centroids[:, 0], stds


def test_kmeans_python():
    x = np.array([[0.1, 0.2], [0.1, 0.4], [0.1, 0.6], [1.0, 0.2], [1.0, 0.1], [1.0, 0.0]])
    kmeans = init_kmeans_python(n_clusters=3)
    centroids, labels = kmeans.fit(X=x)

    print(kmeans.labels_)
    test_set = np.array([[0, 0], [1.2, 0.3]])
    print(kmeans.predict(test_set))
    print(kmeans.cluster_centers_)


if __name__ == "__main__":
    test_kmeans_python()
