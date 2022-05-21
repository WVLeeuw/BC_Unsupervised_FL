import pickle
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
import os

import numpy as np
from scipy.spatial.distance import euclidean

fig_path = f'../logs/plots/'

simulated = [f'mal10_IID_real_{i}' for i in range(1, 101)]
simulated_2 = [f'mal10_IID_rs0_real_{j}' for j in range(1, 101)]

stop_condition = 0.05  # make sure this is equal to what it was during the run to get good results.

# obtain max_rounds for simulated
max_rounds = [0 for i in range(len(simulated))]
for i in range(len(simulated)):
    cur_dir = os.listdir(f'../logs/{simulated[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                if int(f[-3:]) > max_rounds[i]:
                    max_rounds[i] = int(f[-3:])
            elif len(f) == 7:
                if int(f[-2:]) > max_rounds[i]:
                    max_rounds[i] = int(f[-2:])
            elif int(f[-1]) > max_rounds[i]:
                max_rounds[i] = int(f[-1])

# obtain max_rounds for simulated_2
max_rounds_2 = [0 for i in range(len(simulated_2))]
for i in range(len(simulated_2)):
    cur_dir = os.listdir(f'../logs/{simulated_2[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                if int(f[-3:]) > max_rounds_2[i]:
                    max_rounds_2[i] = int(f[-3:])
            elif len(f) == 7:
                if int(f[-2:]) > max_rounds_2[i]:
                    max_rounds_2[i] = int(f[-2:])
            elif int(f[-1]) > max_rounds_2[i]:
                max_rounds_2[i] = int(f[-1])

# to eventually obtain the proportions out of total
max_rounds = max_rounds + max_rounds_2
totals = [sum(max_rounds), sum(max_rounds_2)]

init_centers_simulated = []
for i in range(len(simulated)):
    with open(f'../logs/{simulated[i]}/comm_1/initial_centers.data', 'rb') as file:
        init_centers_simulated.append(pickle.load(file))

avg_distances_init = []
min_distances_init = []
for j in range(len(simulated)):
    avg_distances_init.append(sum([euclidean(init_centers_simulated[j][i], init_centers_simulated[j][i + 1]) for i in
                                   range(len(init_centers_simulated[j]) - 1)]) / len(init_centers_simulated[j]))
    min_distances_init.append(min([euclidean(init_centers_simulated[j][i], init_centers_simulated[j][i + 1]) for i in
                                   range(len(init_centers_simulated[j]) - 1)]))

avg_min_dist_init = np.mean(min_distances_init)

init_centers_simulated_2 = []
for i in range(len(simulated_2)):
    with open(f'../logs/{simulated_2[i]}/comm_1/initial_centers.data', 'rb') as file:
        init_centers_simulated_2.append(pickle.load(file))

avg_distances_init_2 = []
min_distances_init_2 = []
for j in range(len(simulated_2)):
    avg_distances_init_2.append(sum([euclidean(init_centers_simulated_2[j][i], init_centers_simulated_2[j][i + 1]) for i in
                                     range(len(init_centers_simulated_2[j]) - 1)]) / len(init_centers_simulated_2[j]))
    min_distances_init_2.append(min([euclidean(init_centers_simulated_2[j][i], init_centers_simulated_2[j][i + 1]) for i in
                                     range(len(init_centers_simulated_2[j]) - 1)]))

# For each log folder, we should compare the observed delta to min_distances.
# To do this, we log the global_centroids at any given point in time and compare the distance between 'updates'.

# Read the round_info file. If it contains 'deltas', we can obtain the summed distance.
delta_lists = []
avg_dist_lists = []

# Obtain deltas for simulated.
temp = []
done_converging = [0 for i in range(max(max_rounds) + 1)]
logs_converged = [False for j in range(len(simulated))]
for i in range(1, max(max_rounds) + 1):
    avg_dists_this_round = []
    updated_this_round = False
    for j in range(len(simulated)):
        round_info_file = f'../logs/{simulated[j]}/comm_{i}/round_{i}_info.txt'
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
                        if len(re.findall("\d+\.\d+", deltas[el])) > 0:
                            summed_distance += float(re.findall("\d+\.\d+", deltas[el])[0])
                            delta_count += 1
                            delta_lists.append(summed_distance)
                            avg_dists_this_round.append(summed_distance / delta_count)
                    if delta_count > 0:
                        avg_delta = summed_distance/delta_count
                        if avg_delta / min_distances_init[j] < stop_condition and not logs_converged[j]:
                            done_converging[i] += 1  # should only be incremented once per log round
                            logs_converged[j] = True
    if not updated_this_round:
        temp.append(None)
    else:
        temp.append(sum(avg_dists_this_round)/len(avg_dists_this_round))

print(done_converging, np.cumsum(done_converging))
# print(len(delta_lists), min(delta_lists), max(delta_lists))

# Do the same for simulated_2.
delta_lists_2 = []
avg_dist_lists_2 = []

temp_2 = []
done_converging_2 = [0 for i in range(max(max_rounds_2) + 1)]
logs_converged_2 = [False for j in range(len(simulated_2))]
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
                        if len(re.findall("\d+\.\d+", deltas[el])) > 0:
                            summed_distance += float(re.findall("\d+\.\d+", deltas[el])[0])
                            delta_count += 1
                            delta_lists_2.append(summed_distance)
                            avg_dists_this_round.append(summed_distance / delta_count)
                    if delta_count > 0:
                        if (summed_distance / delta_count) / min_distances_init_2[j] < stop_condition and \
                                not logs_converged_2[j]:
                            done_converging_2[i] += 1
                            logs_converged_2[j] = True
    if not updated_this_round:
        temp_2.append(None)
    else:
        temp_2.append(sum(avg_dists_this_round)/len(avg_dists_this_round))

print(done_converging_2, np.cumsum(done_converging_2))
# print(len(delta_lists_2), min(delta_lists_2), max(delta_lists_2))

print(min_distances_init[:5], min_distances_init_2[:5])

# Now build the plot.
fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)

colors = ['purple', 'orange']

rounds = list(range(1, max(max_rounds) + 1))  # for the x-axis
rounds_2 = list(range(1, max(max_rounds_2) + 1))  # idem, for simulated_2
done_converging_cumsum = np.cumsum(done_converging[1:])  # for the y-axis
done_converging_cumsum_2 = np.cumsum(done_converging_2[1:])  # idem, for simulated_2
done_converging_cumsum = [i/len(simulated) for i in done_converging_cumsum]  # then do it as proportion of total
done_converging_cumsum_2 = [i/len(simulated_2) for i in done_converging_cumsum_2]  # idem for simulated_2

ax1.plot(rounds, done_converging_cumsum, linestyle='-', color=colors[0], label='with rep. system')
ax1.plot(rounds_2, done_converging_cumsum_2, linestyle='-', color=colors[1], label='without rep. system')
ax1.set_title('Cumulative runs done converging at each iteration.')
ax1.set_xlabel('iteration')
ax1.set_ylabel('number of runs converged')
ax1.set_ylim([0, 1])
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.legend()

filename = 'cumulative_convergence_mal0_rep_vs_no_rep_nonIID_200rounds.png'
# plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
plt.show()

silhouette_avgs = []
davies_bouldin_scores = []
for i in range(len(simulated)):
    if logs_converged[i]:
        # extract the silhouette avg (and Davies-Bouldin index)
        silhouette_avg = []
        davies_bouldin = []
        for j in range(1, max_rounds[i] + 1):
            round_info_file = f'../logs/{simulated[i]}/comm_{j}/round_{j}_info.txt'
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Combining' in line:
                    if 'silhouette' in line:
                        silhouette_avg.append(float(line.split(sep=':')[-1]))
                    else:
                        davies_bouldin.append(float(line.split(sep=':')[-1]))
        silhouette_avgs.append(silhouette_avg)
        davies_bouldin_scores.append(davies_bouldin)

silhouette_avgs_2 = []
davies_bouldin_scores_2 = []
for j in range(len(simulated_2)):
    if logs_converged_2[j]:
        # idem
        silhouette_avg = []
        davies_bouldin = []
        for k in range(1, max_rounds[j] + 1):
            round_info_file = f'../logs/{simulated[j]}/comm_{k}/round_{k}_info.txt'
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Combining' in line:
                    if 'silhouette' in line:
                        silhouette_avg.append(float(line.split(sep=':')[-1]))
                    else:
                        davies_bouldin.append(float(line.split(sep=':')[-1]))
        silhouette_avgs_2.append(silhouette_avg)
        davies_bouldin_scores_2.append(davies_bouldin)

final_silhouette_scores = [i[-1] for i in silhouette_avgs if len(i) > 0]
final_davies_bouldin_scores = [j[-1] for j in davies_bouldin_scores if len(j) > 0]

print(f"Average silhoutte score across converged runs with rep. system: "
      f"{sum(final_silhouette_scores)/len(final_silhouette_scores)}.")
print(f"Average Davies-Bouldin score across converged runs with rep. system: "
      f"{sum(final_davies_bouldin_scores)/len(final_davies_bouldin_scores)}.")

final_silhouette_scores_2 = [i[-1] for i in silhouette_avgs_2 if len(i) > 0]
final_davies_bouldin_scores_2 = [j[-1] for j in davies_bouldin_scores_2 if len(j) > 0]

print(f"Average silhoutte score across converged runs without rep. system: "
      f"{sum(final_silhouette_scores_2)/len(final_silhouette_scores_2)}.")
print(f"Average Davies-Bouldin score across converged runs without rep. system: "
      f"{sum(final_davies_bouldin_scores_2)/len(final_davies_bouldin_scores_2)}.")

print(len(simulated), len([i for i in logs_converged if i]))
print(len(simulated_2), len([i for i in logs_converged_2 if i]))
