import numpy as np
import random

# This file only contains helper functions for the initialization of the global k-means model.


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
