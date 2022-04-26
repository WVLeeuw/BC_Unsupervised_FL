import pickle
import re

import matplotlib.pyplot as plt
import sys
import os

import numpy as np
from scipy.spatial.distance import euclidean

fig_path = f'../logs/plots/'

simulated_1 = ['mal30_nonIID_blobs_ns1_6', 'mal30_nonIID_blobs_ns1_7', 'mal30_nonIID_blobs_ns1_8', 'mal30_nonIID_blobs_ns1_9', 'mal30_nonIID_blobs_ns1_10']
simulated_2 = ['mal30_nonIID_blobs_ns1_rs0_1', 'mal30_nonIID_blobs_ns1_rs0_2', 'mal30_nonIID_blobs_ns1_rs0_3', 'mal30_nonIID_blobs_ns1_rs0_4', 'mal30_nonIID_blobs_ns1_rs0_5']
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

avg_min_dist_init_1 = np.mean(min_distances_init_1)
avg_min_dist_init_2 = np.mean(min_distances_init_2)

# For each log folder, we should compare the observed delta to min_distances.
# To do this, we log the global_centroids at any given point in time and compare the distance between 'updates'.

# Read the round_info file. If it contains 'deltas', we can obtain the summed distance.
delta_lists_1, delta_lists_2 = [], []
avg_dist_lists_1, avg_dist_lists_2 = [], []

# Obtain deltas for simulated_1.
temp_1 = []
for i in range(1, max(max_rounds_1) + 1):
    avg_dists_this_round = []
    updated_this_round = False
    for j in range(len(simulated_1)):
        round_info_file = f'../logs/{simulated_1[j]}/comm_{i}/round_{i}_info.txt'
        if os.path.exists(round_info_file):
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Deltas' in line:
                    updated_this_round = True
                    summed_distance = 0
                    delta_count = 0
                    deltas = line.split(':')[-1].split(',')
                    for el in range(len(deltas)):
                        summed_distance += float(re.findall("\d+\.\d+", deltas[el])[0])
                        delta_count += 1
                    avg_dists_this_round.append(summed_distance / delta_count)
    if not updated_this_round:
        temp_1.append(None)
    else:
        temp_1.append(sum(avg_dists_this_round)/len(avg_dists_this_round))

# Obtain deltas for simulated_2.
temp_2 = []
for i in range(1, max(max_rounds_2) + 1):
    avg_dists_this_round = []
    updated_this_round = False
    for j in range(len(simulated_2)):
        round_info_file = f'../logs/{simulated_2[j]}/comm_{i}/round_{i}_info.txt'
        if os.path.exists(round_info_file):
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Deltas' in line:
                    updated_this_round = True
                    summed_distance = 0
                    delta_count = 0
                    deltas = line.split(':')[-1].split(',')
                    for el in range(len(deltas)):
                        summed_distance += float(re.findall("\d+\.\d+", deltas[el])[0])
                        delta_count += 1
                    avg_dists_this_round.append(summed_distance / delta_count)
    if not updated_this_round:
        temp_2.append(None)
    else:
        temp_2.append(sum(avg_dists_this_round)/len(avg_dists_this_round))

# Now build the plot.
fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)

colors = ['purple', 'orange']

# Plot the convergence for each log folder.
avg_dists_1 = np.array(temp_1).astype(np.double)
prop_dists_1 = avg_dists_1/avg_min_dist_init_1
cur_mask = np.isfinite(avg_dists_1)
ax1.plot(np.arange(1, max(max_rounds_1) + 1)[cur_mask], prop_dists_1[cur_mask], linestyle='-', marker='o',
         color=colors[0], label='with rep. system')

print(prop_dists_1[cur_mask])

avg_dists_2 = np.array(temp_2).astype(np.double)
prop_dists_2 = avg_dists_2/avg_min_dist_init_2
cur_mask = np.isfinite(avg_dists_2)
ax1.plot(np.arange(1, max(max_rounds_2) + 1)[cur_mask], prop_dists_2[cur_mask], linestyle='-', marker='o',
         color=colors[1], label='without rep. system')

print(prop_dists_2[cur_mask])

ax1.set_title('Convergence over time.')
ax1.set_xlabel('round number')
ax1.set_ylabel('proportional distance w.r.t. distance between initial centers')
ax1.legend()

filename = 'convergence_mal_30_nonIID_blobs_rep_strict_vs_no_rep.png'
plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
plt.show()

