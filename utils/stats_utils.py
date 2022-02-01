import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

project_dir = os.path.dirname(os.getcwd())

# This file describes utility functions related to statistics, such as performance measures.


def evaluate(kmeans, x, splits, use_metric='silhouette', federated=False, verbose=False):
    scores = {}
    centroids = kmeans.cluster_centers_
    for split in splits:
        if federated:
            x[split] = np.concatenate(x[split], axis=0)
        labels = kmeans.predict(x[split])
        if verbose:
            print(split, use_metric)
        if use_metric == "davies_bouldin":
            return NotImplementedError
        elif use_metric == "euclidean":
            return NotImplementedError
        else:
            assert use_metric == 'silhouette'
            score = metrics.silhouette_score(x[split], labels)
        scores[split] = score
        if verbose:
            print(score)
    return scores


def plot_stats(stats, x_variable, x_variable_name, metric_name):
    for spl, spl_dict in stats.items():
        for stat, stat_values in spl_dict.items():
            stats[spl][stat] = np.array(stat_values)

    if x_variable[-1] is None:
        x_variable[-1] = 1
    x_variable = ["single" if i == 0.0 else i for i in x_variable]
    x_axis = np.array(range(len(x_variable)))

    plt.plot(stats['train']['avg'], 'r-', label='Train')
    plt.plot(stats['test']['avg'], 'b-', label='Test')
    plt.fill_between(x_axis,
                     stats['train']['avg'] - stats['train']['std'],
                     stats['train']['avg'] + stats['train']['std'],
                     facecolor='r',
                     alpha=0.3)
    plt.fill_between(x_axis,
                     stats['test']['avg'] - stats['test']['std'],
                     stats['test']['avg'] + stats['test']['std'],
                     facecolor='b',
                     alpha=0.2)
    plt.xticks(x_axis, x_variable)
    plt.xlabel(x_variable_name)
    plt.ylabel(metric_name)
    plt.legend()
    # fig_path = os.path.join(project_dir, "results")
    # plt.savefig(os.path.join(fig_path, "stats_{}.png".format(x_variable_name)), dpi=600, bbox_inches='tight')
    plt.show()
