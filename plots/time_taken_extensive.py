import matplotlib.pyplot as plt
import sys
import os

import numpy as np

fig_path = f'../logs/plots/'

simulated_1 = ['nonIID_real_200rounds_1', 'nonIID_real_200rounds_2', 'nonIID_real_200rounds_3', 'nonIID_real_200rounds_4', 'nonIID_real_200rounds_5']
simulated_2 = ['mal10_nonIID_rs0_real_1', 'mal10_nonIID_rs0_real_2', 'mal10_nonIID_rs0_real_3', 'mal10_nonIID_rs0_real_4', 'mal10_nonIID_rs0_real_5']
entire_log = simulated_1 + simulated_2

# obtain max_rounds for simulated_1
max_rounds_1 = [0 for i in range(len(simulated_1))]
for i in range(len(simulated_1)):
    cur_dir = os.listdir(f'../logs/{simulated_1[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                if int(f[-3:]) > max_rounds_1[i]:
                    max_rounds_1[i] = int(f[-3:])
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
                if int(f[-3:]) > max_rounds_2[i]:
                    max_rounds_2[i] = int(f[-3:])
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
avg_est_excl_role_1, avg_est_excl_role_2 = [], []
times_per_log_1, times_per_log_2 = [0 for i in range(len(simulated_1))], [0 for i in range(len(simulated_2))]
# Obtain values for simulated_1
for i in range(1, max(max_rounds_1) + 1):
    times_this_round = []
    est_times_this_round = []
    est_excl_role_this_round = []
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
                    times_per_log_1[j] += float(line.split(sep=':')[-1].split()[0])
                if 'Estimate time without' in line:
                    est_excl_role_this_round.append(float(line.split(sep=':')[-1].split()[0]))
    avg_time_spent_1.append(sum(times_this_round)/len(times_this_round))
    avg_est_time_spent_1.append(sum(est_times_this_round)/len(est_times_this_round))
    avg_est_excl_role_1.append(sum(est_excl_role_this_round)/len(est_excl_role_this_round))

# Obtain values for simulated_2
for i in range(1, max(max_rounds_2) + 1):
    times_this_round = []
    est_times_this_round = []
    est_excl_role_this_round = []
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
                    times_per_log_2[j] += float(line.split(sep=':')[-1].split()[0])
                if 'Estimate time without' in line:
                    est_excl_role_this_round.append(float(line.split(sep=':')[-1].split()[0]))
    avg_time_spent_2.append(sum(times_this_round)/len(times_this_round))
    avg_est_time_spent_2.append(sum(est_times_this_round)/len(est_times_this_round))
    avg_est_excl_role_2.append(sum(est_excl_role_this_round) / len(est_excl_role_this_round))

avg_per_run_1 = sum(avg_time_spent_1)/max(max_rounds_1) * (sum(max_rounds_1)/len(max_rounds_1))
avg_per_run_2 = sum(avg_time_spent_2)/max(max_rounds_2) * (sum(max_rounds_2)/len(max_rounds_2))
per_100_1 = sum(avg_time_spent_1)/max(max_rounds_1) * 100
per_100_2 = sum(avg_time_spent_2)/max(max_rounds_2) * 100
avg_est_per_run_1 = np.mean(np.divide(times_per_log_1, max_rounds_1)) * (sum(max_rounds_1)/len(max_rounds_1))
avg_est_per_run_2 = np.mean(np.divide(times_per_log_2, max_rounds_2)) * (sum(max_rounds_2)/len(max_rounds_2))
est_per_100_1 = np.mean(np.divide(times_per_log_1, max_rounds_1)) * 100
est_per_100_2 = np.mean(np.divide(times_per_log_2, max_rounds_2)) * 100
avg_est_excl_role_per_run_1 = sum(avg_est_excl_role_1)/max(max_rounds_1) * (sum(max_rounds_1)/len(max_rounds_1))
avg_est_excl_role_per_run_2 = sum(avg_est_excl_role_2)/max(max_rounds_1) * (sum(max_rounds_2)/len(max_rounds_2))
est_excl_role_per_100_1 = sum(avg_est_excl_role_1)/max(max_rounds_1) * 100
est_excl_role_per_100_2 = sum(avg_est_excl_role_2)/max(max_rounds_2) * 100
print(f'Average total time per run is {avg_per_run_1} and {avg_per_run_2} seconds respectively, '
      f'or {per_100_1} and {per_100_2} seconds per 100 rounds respectively.')
print(f'Average total time estimate per run is {avg_est_per_run_1} and {avg_est_per_run_2} seconds respectively, '
      f'or {est_per_100_1} and {est_per_100_2} seconds per 100 rounds respectively.')
print(f'Excluding role assignment, the average total time estimate per run is {avg_est_excl_role_per_run_1} and '
      f'{avg_est_excl_role_per_run_2} seconds respectively, or {est_excl_role_per_100_1} and {est_excl_role_per_100_2} '
      f'per 100 rounds respectively.')

# Now build the plots.
colors = ['purple', 'orange']

fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)
ax1.plot(range(1, max(max_rounds_1) + 1), avg_time_spent_1, linestyle='-', color=colors[0])
ax1.plot(range(1, max(max_rounds_2) + 1), avg_time_spent_2, linestyle='-', color=colors[1])
ax1.set_title('Average time spent per learning round.')
ax1.set_ylabel('Time spent (s)')
ax1.set_xlabel('Round number')
ax1.set_ylim([0, 4])
ax1.legend(['with rep. system', 'without rep. system'])

plt.show()

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.subplots(1, 1)
ax2.plot(range(1, max(max_rounds_1) + 1), avg_est_time_spent_1, linestyle='-', color=colors[0])
ax2.plot(range(1, max(max_rounds_2) + 1), avg_est_time_spent_2, linestyle='-', color=colors[1])
ax2.set_title('Average estimated time spent per learning round if run in parallel.')
ax2.set_ylabel('Time spent (s)')
ax2.set_xlabel('Round number')
ax2.set_ylim([0, 4])
ax2.legend(['with rep. system', 'without rep. system'])

filename = 'time_taken_comparison_mal20_rep_vs_norep_IID_200rounds.png'
plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')

plt.show()

fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.subplots(1, 1)
ax3.plot(range(1, max(max_rounds_1) + 1), avg_est_excl_role_1, linestyle='-', color=colors[0])
ax3.plot(range(1, max(max_rounds_2) + 1), avg_est_excl_role_2, linestyle='-', color=colors[1])
ax3.set_title('Average estimated time spent per learning round in parallel, excluding role assignment.')
ax3.set_ylabel('Time spent (s)')
ax3.set_xlabel('Round number')
ax3.set_ylim([0, 2])
ax3.legend(['with rep. system', 'without rep. system'])

filename = 'est_time_taken_wo_role_assign_comparison_mal20_rep_vs_norep_IID_200rounds.png'
plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')

plt.show()
