import matplotlib.pyplot as plt
import sys
import os

import numpy as np

log_folders = ['closely_real_1', 'closely_real_2', 'closely_real_3', 'closely_real_4', 'closely_real_5']
fig_path = f'../logs/plots/'

max_rounds = [0 for i in range(len(log_folders))]
for i in range(len(log_folders)):
    cur_dir = os.listdir(f'../logs/{log_folders[i]}')
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

total_rounds = sum(max_rounds)  # used to get the eventual proportions

assert len(max_rounds) == len(log_folders), \
    'Could not find the number of communication rounds for (at least) one of the supplied runs.'

# Now, instead of counting all regardless of file, we want a list of counts per file.
role_counts = []
# Each element contains two lists, one representing the counts of benign devices, the other representing the counts
# of malicious devices.
for i in range(len(log_folders)):
    B_none_count, M_none_count = 0, 0
    B_data_owner_count, M_data_owner_count = 0, 0
    B_committee_count, M_committee_count = 0, 0
    B_leader_count, M_leader_count = 0, 0
    all_rounds = [f'comm_{i}' for i in range(1, max_rounds[i] + 1)]
    for round_iter in all_rounds:
        devices_assign_file = f'../logs/{log_folders[i]}/{round_iter}/role_assignment_{round_iter}.txt'
        file = open(devices_assign_file, 'r')
        lines_list = file.read().split('\n')
        for line in lines_list:
            if 'M' in line:
                if 'None' in line:
                    M_none_count += 1
                if 'data owner' in line:
                    M_data_owner_count += 1
                if 'committee' in line:
                    M_committee_count += 1
                if 'leader' in line:
                    M_leader_count += 1
            else:  # must be a regular device.
                if 'None' in line:
                    B_none_count += 1
                if 'data owner' in line:
                    B_data_owner_count += 1
                if 'committee' in line:
                    B_committee_count += 1
                if 'leader' in line:
                    B_leader_count += 1
    role_counts.append([[B_none_count, B_data_owner_count, B_committee_count, B_leader_count],
                        [M_none_count, M_data_owner_count, M_committee_count, M_leader_count]])

B_counts = [i[0] for i in role_counts]  # first element of each list in role_counts
M_counts = [i[-1] for i in role_counts]  # last element of each list in role_counts
# total_devices_per_folder has a list for each log folder, first element being the no. benign devices,
# second being the no. malicious devices.
total_devices_per_folder = [[sum(B_counts[i]), sum(M_counts[i])] for i in range(len(max_rounds))]

props_benign_per_folder = [np.array(B_counts[i])/total_devices_per_folder[i][0] for i in range(len(total_devices_per_folder))]
props_malicious_per_folder = [np.array(M_counts[i])/total_devices_per_folder[i][-1] for i in range(len(total_devices_per_folder))]

# Sanity check
print(len(props_benign_per_folder), len(props_malicious_per_folder))
print(len(props_benign_per_folder[0]), len(props_malicious_per_folder[0]))

# Now build the figure.
fig = plt.figure(figsize=(8, 8))
ax1, ax2 = fig.subplots(2, 1, sharex=True)

labels = ['None', 'data owner', 'committee', 'leader']
colors = ['green', 'purple', 'orange', 'magenta', 'cyan']

n = len(labels)
r = np.arange(n)
width = .15

for i in range(len(log_folders)):
    ax1.bar(r + width * (i - 1), props_malicious_per_folder[i], color=colors[i], width=width, edgecolor='k',
            label=log_folders[i])
ax1.set_title('Role assignment for malicious devices')
ax1.set_ylabel('proportion of rounds assigned')
ax1.set_ylim([0, 1])

for i in range(len(log_folders)):
    ax2.bar(r + width * (i - 1), props_benign_per_folder[i], color=colors[i], width=width, edgecolor='k',
            label=log_folders[i])
ax2.set_title('Role assignment for regular devices')
ax2.set_ylabel('proportion of rounds assigned')
ax2.set_ylim([0, 1])
plt.xticks(r + width/2, labels)

fig.legend(log_folders, loc='upper right')
filename = 'role_assignment_closely_real_per_folder.png'
# plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
plt.show()
