import pickle
import re

import matplotlib.pyplot as plt
import sys
import os

import numpy as np
from scipy.spatial.distance import euclidean

simulated_1 = ['04122022_115042']
simulated_2 = ['04122022_115114']
entire_log = simulated_1 + simulated_2

# obtain max_rounds for simulated_1
max_rounds_1 = [0 for i in range(len(simulated_1))]
for i in range(len(simulated_1)):
    cur_dir = os.listdir(f'../logs/{simulated_1[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                max_rounds_1[i] = 100
            elif len(f) == 7:
                if int(f[-2:]) > max_rounds_1[i]:
                    max_rounds_1[i] = int(f[-2:])
            elif int(f[-1]) > max_rounds_1[i]:
                max_rounds_1[i] = int(f[-1])

# idem for simulated_2
max_rounds_2 = [0 for i in range(len(simulated_2))]
for i in range(len(simulated_2)):
    cur_dir = os.listdir(f'../logs/{simulated_2[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                max_rounds_2[i] = 100
            elif len(f) == 7:
                if int(f[-2:]) > max_rounds_2[i]:
                    max_rounds_2[i] = int(f[-2:])
            elif int(f[-1]) > max_rounds_2[i]:
                max_rounds_2[i] = int(f[-1])

# to eventually obtain the proportions out of total
max_rounds = max_rounds_1 + max_rounds_2
totals = [sum(max_rounds_1), sum(max_rounds_2)]

assert len(max_rounds_1 + max_rounds_2) == len(entire_log), \
    'Could not find the number of communication rounds for (at least) one of the supplied runs.'

init_centers_simulated_1, init_centers_simulated_2 = [], []
for i in range(len(simulated_1)):
    with open(f'../logs/{simulated_1[i]}/comm_1/initial_centers.data', 'rb') as file:
        init_centers_simulated_1.append(pickle.load(file))

for i in range(len(simulated_2)):
    with open(f'../logs/{simulated_2[i]}/comm_1/initial_centers.data', 'rb') as file:
        init_centers_simulated_2.append(pickle.load(file))

avg_distances_init_1, avg_distances_init_2 = [], []
min_distances_init_1, min_distances_init_2 = [], []
for j in range(len(simulated_1)):
    avg_distances_init_1.append(sum([euclidean(init_centers_simulated_1[j][i], init_centers_simulated_1[j][i + 1]) for i in
                                     range(len(init_centers_simulated_1[j]) - 1)]) / len(init_centers_simulated_1[j]))
    min_distances_init_1.append(min([euclidean(init_centers_simulated_1[0][i], init_centers_simulated_1[0][i + 1]) for i in
                                     range(len(init_centers_simulated_1[0]) - 1)]))

for j in range(len(simulated_2)):
    avg_distances_init_2.append(sum([euclidean(init_centers_simulated_2[j][i], init_centers_simulated_2[j][i + 1]) for i in
                                     range(len(init_centers_simulated_2[j]) - 1)]) / len(init_centers_simulated_2[j]))
    min_distances_init_2.append(min([euclidean(init_centers_simulated_2[0][i], init_centers_simulated_2[0][i + 1]) for i in
                                     range(len(init_centers_simulated_2[0]) - 1)]))

# For each log folder, we should compare the observed delta to min_distances.
# To do this, we log the global_centroids at any given point in time and compare the distance between 'updates'.

# Read the round_info file. If it contains 'deltas', we can obtain the summed distance.
delta_lists_1, delta_lists_2 = [], []
avg_dist_lists_1, avg_dist_lists_2 = [], []

# Obtain deltas for simulated_1.
for i in range(len(simulated_1)):
    delta_list = []
    avg_dist_list = []
    for j in range(1, max_rounds_1[i] + 1):
        # check whether there are deltas
        round_info_file = f'../logs/{simulated_1[i]}/comm_{j}/round_{j}_info.txt'
        file = open(round_info_file, 'r')
        lines_list = file.read().split('\n')
        updated_this_round = False
        for line in lines_list:
            if 'Deltas' in line:
                updated_this_round = True
                summed_distance = 0
                delta_count = 0
                deltas = line.split(':')[-1].split(',')
                for el in range(len(deltas)):
                    summed_distance += float(re.findall("\d+\.\d+", deltas[el])[0])
                    delta_count += 1
                delta_list.append(summed_distance)
                avg_dist_list.append(summed_distance / delta_count)

        if not updated_this_round:
            delta_list.append(None)
            avg_dist_list.append(None)

    delta_lists_1.append(delta_list)
    avg_dist_lists_1.append(avg_dist_list)

# Obtain deltas for simulated_2.
for i in range(len(simulated_2)):
    delta_list = []
    avg_dist_list = []
    for j in range(1, max_rounds_2[i] + 1):
        # check whether there are deltas
        round_info_file = f'../logs/{simulated_2[i]}/comm_{j}/round_{j}_info.txt'
        file = open(round_info_file, 'r')
        lines_list = file.read().split('\n')
        updated_this_round = False
        for line in lines_list:
            if 'Deltas' in line:
                updated_this_round = True
                summed_distance = 0
                delta_count = 0
                deltas = line.split(':')[-1].split(',')
                for el in range(len(deltas)):
                    summed_distance += float(re.findall("\d+\.\d+", deltas[el])[0])
                    delta_count += 1
                delta_list.append(summed_distance)
                avg_dist_list.append(summed_distance / delta_count)

        if not updated_this_round:
            delta_list.append(None)
            avg_dist_list.append(None)

    delta_lists_2.append(delta_list)
    avg_dist_lists_2.append(avg_dist_list)

# Now build the plot.
fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)

colors = ['purple', 'orange']

# Plot the convergence for each log folder.
for i in range(len(simulated_1)):
    min_init_dist = np.array(min_distances_init_1[i]).astype(np.double)
    avg_init_dist = np.array(avg_distances_init_1[i]).astype(np.double)
    cur_avg_dists = np.array(avg_dist_lists_1[i]).astype(np.double)
    cur_prop_dists = cur_avg_dists/min_init_dist
    cur_mask = np.isfinite(cur_avg_dists)
    ax1.plot(np.arange(1, max_rounds_1[i] + 1)[cur_mask], cur_prop_dists[cur_mask], linestyle='-', marker='o',
             color=colors[0], label=simulated_1[i])

for i in range(len(simulated_2)):
    min_init_dist = np.array(min_distances_init_2[i]).astype(np.double)
    avg_init_dist = np.array(avg_distances_init_2[i]).astype(np.double)
    cur_avg_dists = np.array(avg_dist_lists_2[i]).astype(np.double)
    cur_prop_dists = cur_avg_dists / min_init_dist
    cur_mask = np.isfinite(cur_avg_dists)
    ax1.plot(np.arange(1, max_rounds_2[i] + 1)[cur_mask], cur_prop_dists[cur_mask], linestyle='-', marker='o',
             color=colors[1], label=simulated_2[i])

ax1.set_title('Convergence over time.')
ax1.set_xlabel('round number')
ax1.set_ylabel('proportional distance w.r.t. distance between initial centers')
ax1.legend(['IID', 'non-IID'])

plt.show()

