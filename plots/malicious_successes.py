import matplotlib.pyplot as plt
import sys
import os
import block
from blockchain import Blockchain
import json

log_folders = ['04042022_102601', '04042022_103057', '04042022_104125']

# Plot, for the provided log folder, the rounds (i.e. scatter) in which a malicious leader managed to have
# their block added to the chain.
max_rounds = [0 for i in range(len(log_folders))]
for i in range(len(log_folders)):
    cur_dir = os.listdir(f'../logs/{log_folders[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) == 6:
                max_rounds[i] = int(f[-1])
            elif len(f) == 7:
                max_rounds[i] = int(f[-2:])
            else:
                max_rounds[i] = 100

# Obtain the malicious devices from malicious_devices_comm_i.txt
all_mal_devices = []
indices_success_per_folder = []
indices_leader_per_folder = []

for i in range(len(log_folders)):
    mal_devices = []
    malicious_devices_file = f'../logs/{log_folders[i]}/comm_1/malicious_devices_comm_1.txt'
    devices_file = open(malicious_devices_file, 'r')
    lines = devices_file.read().split('\n')
    for line in lines:
        if len(line) > 0:
            mal_devices.append(line.split()[0])
    all_mal_devices.append(mal_devices)

    # Obtain the round nrs at which a malicious device was assigned 'leader'.
    indices_mal_leader = []
    # Obtain the round nrs at which a malicious leader won.
    indices_mal_winner = []
    for j in range(1, max_rounds[i] + 1):
        round_info_file = f'../logs/{log_folders[i]}/comm_{j}/round_{j}_info.txt'
        info_file = open(round_info_file, 'r')
        lines = info_file.read().split('\n')
        for line in lines:
            if 'alignment' in line:
                if 'M' in line:
                    indices_mal_winner.append(j)

        role_assignment_file = f'../logs/{log_folders[i]}/comm_{j}/malicious_devices_comm_{j}.txt'
        role_file = open(role_assignment_file, 'r')
        role_lines = role_file.read().split('\n')
        for line in role_lines:
            if 'leader' in line:
                indices_mal_leader.append(j)

    indices_success_per_folder.append(indices_mal_winner)
    indices_leader_per_folder.append(indices_mal_leader)

# fig, ax = plt.subplots(1, 1)
# colors = ['purple', 'orange', 'magenta', 'green', 'cyan']
#
# for i in range(len(indices_success_per_folder)):
#     ax.plot(indices_success_per_folder[i], [log_folders[i] for j in range(len(indices_success_per_folder[i]))],
#             marker='o', color=colors[i], mfc='none', linestyle='None')
#
# ax.set_title('Round numbers corresponding with malicious leader successes.')
# # ax.set_ylabel('Log folder')
# ax.set_yticks([])
# ax.set_xlabel('Round numbers')
# ax.set_xlim([0, max(max_rounds) + 1])
# ax.legend(log_folders)
# plt.show()

total_mal_devices = sum([len(i) for i in all_mal_devices])
avg_mal_devices = total_mal_devices/len(log_folders)

# Calculate proportion of time a malicious device was leader and the proportion of time they won if they were.
total_mal_leaders = sum([len(i) for i in indices_leader_per_folder])
prop_mal_leader = (total_mal_leaders/sum(max_rounds))/avg_mal_devices
total_mal_won = sum([len(i) for i in indices_success_per_folder])
prop_mal_won = total_mal_won/total_mal_leaders

print('Malicious devices won ' + ('%.2f' % (prop_mal_won * 100)) + '% of the time they were assigned to be a leader.')
print('Malicious devices had a ' + ('%.2f' % (prop_mal_leader * 100)) + '% chance to be assigned leader.')
