import matplotlib.pyplot as plt
import sys
import os

import numpy as np

fig_path = f'../logs/plots/'

simulated_1 = ['IID_real_1', 'IID_real_2', 'IID_real_3', 'IID_real_4', 'IID_real_5']
simulated_2 = ['nonIID_real_1', 'nonIID_real_2', 'nonIID_real_3', 'nonIID_real_4', 'nonIID_real_5']
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

print(max_rounds)  # sanity check

avg_time_spent_1, avg_time_spent_2 = [], []
avg_est_time_spent_1, avg_est_time_spent_2 = [], []
# Obtain values for simulated_1
for i in range(1, max(max_rounds_1) + 1):
    times_this_round = []
    est_times_this_round = []
    for j in range(len(simulated_1)):
        round_info_file = f'../logs/{simulated_1[j]}/comm_{i}/round_{i}_info.txt'
        if os.path.exists(round_info_file):
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Time spent' in line:
                    times_this_round.append(float(line.split(sep=':')[-1].split()[0]))
                if 'Estimate time taken' in line:
                    est_times_this_round.append(float(line.split(sep=':')[-1].split()[0]))
    avg_time_spent_1.append(sum(times_this_round)/len(times_this_round))
    avg_est_time_spent_1.append(sum(est_times_this_round)/len(est_times_this_round))

print(f'Average total time per run is {sum(avg_time_spent_1)/max(max_rounds) * 100} seconds.')
print(f'Average total time estimate per run is {sum(avg_est_time_spent_1)/max(max_rounds) * 100} seconds.')

# Obtain values for simulated_2
for i in range(1, max(max_rounds_2) + 1):
    times_this_round = []
    est_times_this_round = []
    for j in range(len(simulated_2)):
        round_info_file = f'../logs/{simulated_2[j]}/comm_{i}/round_{i}_info.txt'
        if os.path.exists(round_info_file):
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Time spent' in line:
                    times_this_round.append(float(line.split(sep=':')[-1].split()[0]))
                if 'Estimate time taken' in line:
                    est_times_this_round.append(float(line.split(sep=':')[-1].split()[0]))
    avg_time_spent_2.append(sum(times_this_round)/len(times_this_round))
    avg_est_time_spent_2.append(sum(est_times_this_round)/len(est_times_this_round))

print(f'Average total time per run is {sum(avg_time_spent_2)/max(max_rounds) * 100} seconds.')
print(f'Average total time estimate per run is {sum(avg_est_time_spent_2)/max(max_rounds) * 100} seconds.')

# Now build the plots.
colors = ['purple', 'orange']

fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)
ax1.plot(range(1, max(max_rounds_1) + 1), avg_time_spent_1, linestyle='-', marker='o', color=colors[0])
ax1.plot(range(1, max(max_rounds_2) + 1), avg_time_spent_2, linestyle='-', marker='o', color=colors[1])
ax1.set_title('Average time spent per learning round.')
ax1.set_ylabel('Time spent (s)')
ax1.set_xlabel('Round number')
ax1.set_ylim([0, 4])
ax1.legend(['IID', 'non-IID'])

plt.show()

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.subplots(1, 1)
ax2.plot(range(1, max(max_rounds_1) + 1), avg_est_time_spent_1, linestyle='-', marker='o', color=colors[0])
ax2.plot(range(1, max(max_rounds_2) + 1), avg_est_time_spent_2, linestyle='-', marker='o', color=colors[1])
ax2.set_title('Average estimated time spent per learning round if run in parallel.')
ax2.set_ylabel('Time spent (s)')
ax2.set_xlabel('Round number')
ax2.set_ylim([0, 4])
ax2.legend(['IID', 'non-IID'])

# filename = 'time_taken_loosely_real_IID.png'
# plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')

plt.show()
