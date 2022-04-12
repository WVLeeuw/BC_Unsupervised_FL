import pickle

import matplotlib.pyplot as plt
import sys
import os
import re

import numpy as np
from scipy.spatial.distance import euclidean

fig_path = f'../logs/plots/'

log_folders = ['IID_1', 'nonIID_1']
fig_path = f'../logs/plots/'

max_rounds = [0 for i in range(len(log_folders))]
for i in range(len(log_folders)):
    cur_dir = os.listdir(f'../logs/{log_folders[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                max_rounds[i] = 100
            elif len(f) == 7:
                if int(f[-2:]) > max_rounds[i]:
                    max_rounds[i] = int(f[-2:])
            elif int(f[-1]) > max_rounds[i]:
                max_rounds[i] = int(f[-1])

total_rounds = sum(max_rounds)  # used to get the eventual proportions

assert len(max_rounds) == len(log_folders), \
    'Could not find the number of communication rounds for (at least) one of the supplied runs.'

init_centers_per_folder = []
for i in range(len(log_folders)):
    with open(f'../logs/{log_folders[i]}/comm_1/initial_centers.data', 'rb') as file:
        init_centers_per_folder.append(pickle.load(file))

print(len(init_centers_per_folder), len(init_centers_per_folder[0]))

avg_distances_init = []
min_distances_init = []
for j in range(len(log_folders)):
    avg_distances_init.append(sum([euclidean(init_centers_per_folder[j][i], init_centers_per_folder[j][i + 1]) for i in
                                   range(len(init_centers_per_folder[j]) - 1)]) / len(init_centers_per_folder[j]))
    min_distances_init.append(min([euclidean(init_centers_per_folder[0][i], init_centers_per_folder[0][i + 1]) for i in
                                   range(len(init_centers_per_folder[0]) - 1)]))
print(len(avg_distances_init), len(min_distances_init))  # should be same as no. log folders

# For each log folder, we should compare the observed delta to min_distances.
# To do this, we log the global_centroids at any given point in time and compare the distance between 'updates'.

# Read the round_info file. If it contains 'deltas', we can obtain the summed distance.
delta_lists = []
avg_dist_lists = []
for i in range(len(log_folders)):
    delta_list = []
    avg_dist_list = []
    for j in range(1, max_rounds[i] + 1):
        # check whether there are deltas
        round_info_file = f'../logs/{log_folders[i]}/comm_{j}/round_{j}_info.txt'
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

    delta_lists.append(delta_list)
    avg_dist_lists.append(avg_dist_list)

print(len(delta_lists), len(delta_lists[0]))  # sanity check

# Now build the plot.
fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)

colors = ['green', 'purple', 'orange', 'magenta', 'cyan']

# Plot the convergence for each log folder.
for i in range(len(log_folders)):
    cur_deltas = np.array(delta_lists[i]).astype(np.double)
    cur_mask = np.isfinite(cur_deltas)
    ax1.plot(np.arange(1, max_rounds[i] + 1)[cur_mask], cur_deltas[cur_mask], linestyle='-', marker='o',
             color=colors[i], label=log_folders[i])
ax1.set_title('Convergence over time.')
ax1.set_xlabel('round number')
ax1.set_ylabel('summed Euclidean distance between updates')

fig.legend(log_folders)
plt.show()

# Build the plot showing convergence proportional to distance between initial centers.
fig1 = plt.figure(figsize=(8, 6))
ax2 = fig1.subplots(1, 1)

for i in range(len(log_folders)):
    min_init_dist = np.array(min_distances_init[i]).astype(np.double)
    avg_init_dist = np.array(avg_distances_init[i]).astype(np.double)
    cur_avg_dists = np.array(avg_dist_lists[i]).astype(np.double)
    cur_prop_dists = cur_avg_dists/min_init_dist
    cur_mask = np.isfinite(cur_avg_dists)
    ax2.plot(np.arange(1, max_rounds[i] + 1)[cur_mask], cur_prop_dists[cur_mask], linestyle='-', marker='o',
             color=colors[i], label=log_folders[i])
ax2.set_title('Convergence over time.')
ax2.set_xlabel('round number')
ax2.set_ylabel('proportional distance w.r.t. distance between initial centers')

fig1.legend(log_folders)
# filename = 'role_assignment_comparison_closely_somewhat.png'
# plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
plt.show()
