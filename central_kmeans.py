from sklearn import cluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import load_iris, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import data_utils

start_time = time.time()

# ToDo: run centralized k-means for multiple datasets.
# X, y = make_blobs(n_samples=500, centers=3, n_features=2, random_state=42)
X, _ = data_utils.create_dummy_data(dims=2, clients_per_cluster=1, clusters=3, samples_each=160)
X = [item for sublist in X for item in sublist]
X = np.asarray(X)

n_clusters = 3
central_model = cluster.KMeans(n_clusters=n_clusters, max_iter=100)
cluster_labels = central_model.fit_predict(X)

centroids = central_model.cluster_centers_
silhouette = silhouette_score(X, cluster_labels)
print(centroids, silhouette)

end_time = time.time() - start_time
print(f"Time taken to train centralized k-means on synthetic data: {end_time} seconds.")

# Silhouette plot
fig, ax1 = plt.subplots(1, 1)
ax1.set_xlim([-0.4, 1])
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
silhouette_avg = silhouette_score(X, cluster_labels)
silhouette_samples = silhouette_samples(X, cluster_labels)

y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette = silhouette_samples[cluster_labels == i]
    ith_cluster_silhouette.sort()

    ith_cluster_size = ith_cluster_silhouette.shape[0]
    y_upper = y_lower + ith_cluster_size

    color = cm.nipy_spectral(float(i) / n_clusters)
    print(y_lower, y_upper)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0,
                      ith_cluster_silhouette,
                      facecolor=color,
                      edgecolor=color,
                      alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * ith_cluster_size, str(i))

    # compute y_lower for next plot
    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color='red', linestyle='--')  # vertical line to show silhouette avg
ax1.set_yticks([])  # no ticks on y-axis
ax1.set_xticks([-.4, -.2, 0, .2, .4, .6, .8, 1])
fig.savefig(fname='figures/silhouette_analysis_centralized_kmeans.png')
plt.show()

# Cluster plot
fig, ax2 = plt.subplots(1, 1)
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=.7, c=colors, edgecolor='k')  # data points
ax2.scatter(centroids[:, 0], centroids[:, 1], marker='o', c="white", s=200, edgecolor='k')  # centers

for i, c in enumerate(centroids):
    ax2.scatter(c[0], c[1], marker="$%d$" % i, s=50, edgecolor='k')

ax2.set_title("The visualization of clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")
fig.savefig(fname='figures/centralized_clustering_result.png')

plt.show()
