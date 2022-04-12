import matplotlib.pyplot as plt
import sys
import os

import numpy as np

fig_path = f'../logs/plots/'

simulated_1 = ['nonIID_1', 'nonIID_2', 'nonIID_3', 'nonIID_4', 'nonIID_5']
simulated_2 = ['mal_10_rs0_1', 'mal_10_rs0_2', 'mal_10_rs0_3', 'mal_10_rs0_4', 'mal_10_rs0_5']
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

B_none_count_1, B_none_count_2, M_none_count_1, M_none_count_2 = 0, 0, 0, 0
B_data_owner_count_1, B_data_owner_count_2, M_data_owner_count_1, M_data_owner_count_2 = 0, 0, 0, 0
B_committee_count_1, B_committee_count_2, M_committee_count_1, M_committee_count_2 = 0, 0, 0, 0
B_leader_count_1, B_leader_count_2, M_leader_count_1, M_leader_count_2 = 0, 0, 0, 0
for i in range(len(entire_log)):
    all_rounds = [f'comm_{i}' for i in range(1, max_rounds[i] + 1)]
    for round_iter in all_rounds:
        devices_assign_file = f'../logs/{entire_log[i]}/{round_iter}/role_assignment_{round_iter}.txt'
        file = open(devices_assign_file, 'r')
        lines_list = file.read().split('\n')
        for line in lines_list:
            if 'M' in line:
                if 'None' in line:
                    if entire_log[i] in simulated_1:
                        M_none_count_1 += 1
                    else:
                        M_none_count_2 += 1
                if 'data owner' in line:
                    if entire_log[i] in simulated_1:
                        M_data_owner_count_1 += 1
                    else:
                        M_data_owner_count_2 += 1
                if 'committee' in line:
                    if entire_log[i] in simulated_1:
                        M_committee_count_1 += 1
                    else:
                        M_committee_count_2 += 1
                if 'leader' in line:
                    if entire_log[i] in simulated_1:
                        M_leader_count_1 += 1
                    else:
                        M_leader_count_2 += 1
            else:  # must be a regular device.
                if 'None' in line:
                    if entire_log[i] in simulated_1:
                        B_none_count_1 += 1
                    else:
                        B_none_count_2 += 1
                if 'data owner' in line:
                    if entire_log[i] in simulated_1:
                        B_data_owner_count_1 += 1
                    else:
                        B_data_owner_count_2 += 1
                if 'committee' in line:
                    if entire_log[i] in simulated_1:
                        B_committee_count_1 += 1
                    else:
                        B_committee_count_2 += 1
                if 'leader' in line:
                    if entire_log[i] in simulated_1:
                        B_leader_count_1 += 1
                    else:
                        B_leader_count_2 += 1

num_malicious_1 = sum([M_none_count_1, M_data_owner_count_1, M_committee_count_1, M_leader_count_1]) / totals[0]
num_malicious_2 = sum([M_none_count_2, M_data_owner_count_2, M_committee_count_2, M_leader_count_2]) / totals[1]
num_regular_1 = sum([B_none_count_1, B_data_owner_count_1, B_committee_count_1, B_leader_count_1]) / totals[0]
num_regular_2 = sum([B_none_count_2, B_data_owner_count_2, B_committee_count_2, B_leader_count_2]) / totals[1]

malicious_counts_1 = [M_none_count_1, M_data_owner_count_1, M_committee_count_1, M_leader_count_1]
regular_counts_1 = [B_none_count_1, B_data_owner_count_1, B_committee_count_1, B_leader_count_1]
malicious_counts_2 = [M_none_count_2, M_data_owner_count_2, M_committee_count_2, M_leader_count_2]
regular_counts_2 = [B_none_count_2, B_data_owner_count_2, B_committee_count_2, B_leader_count_2]

malicious_props_1 = [i/(num_malicious_1*totals[0]) for i in malicious_counts_1]
malicious_props_2 = [i/(num_malicious_2*totals[1]) for i in malicious_counts_2]
regular_props_1 = [i/(num_regular_1*totals[0]) for i in regular_counts_1]
regular_props_2 = [i/(num_regular_2*totals[0]) for i in regular_counts_2]

labels = ['None', 'data owner', 'committee', 'leader']
colors = ['purple', 'orange']

fig = plt.figure(figsize=(8, 8))
ax1, ax2 = fig.subplots(2, 1, sharex=True)

n = len(labels)
r = np.arange(n)
width = .3

ax1.bar(r, malicious_props_1, color=colors[0], width=width, edgecolor='k', label='closely_real')
ax1.bar(r + width, malicious_props_2, color=colors[1], width=width, edgecolor='k', label='somewhat_real')
ax1.set_title('Role assignment for malicious devices')
ax1.set_ylabel('proportion of rounds assigned')
ax1.set_ylim([0, 1])

ax2.bar(r, regular_props_1, color=colors[0], width=width, edgecolor='k', label='closely_real')
ax2.bar(r + width, regular_props_2, color=colors[1], width=width, edgecolor='k', label='somewhat_real')
ax2.set_title('Role assignment for regular devices')
ax2.set_ylabel('proportion of rounds assigned')
ax2.set_ylim([0, 1])
plt.xticks(r + width/2, labels)

fig.legend(['with rep. system', 'without rep. system'])
filename = 'role_assignment_comparison_closely_somewhat.png'
# plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
plt.show()
