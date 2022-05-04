import matplotlib.pyplot as plt
import sys
import os
import block
from blockchain import Blockchain
import json
import ast

log_folders = ['nonIID_blobs_1', 'nonIID_blobs_2', 'nonIID_blobs_3', 'nonIID_blobs_4', 'nonIID_blobs_5']
fig_path = f'../logs/plots/'

# Plot, for the provided log folder, the rounds (i.e. scatter) in which a malicious leader managed to have
# their block added to the chain.
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
        if len(line) > 0 and 'blacklisted' not in line:
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

blacklisted_devices = []  # to read blacklisted nodes
# read for each round which devices were blacklisted.
for i in range(len(log_folders)):
    blacklisted_per_folder = []
    for j in range(1, max_rounds[i] + 1):
        malicious_devices_file = f'../logs/{log_folders[i]}/comm_{j}/malicious_devices_comm_{j}.txt'
        devices_file = open(malicious_devices_file, 'r')
        lines = [line.rstrip() for line in devices_file]
        blacklisted = []
        for line in lines:
            if 'blacklisted' in line:
                blacklisted = [el[1:] for el in line.split(sep=':')[-1].split(',')]
        blacklisted_per_folder.append(blacklisted)
    blacklisted_devices.append(blacklisted_per_folder)

# evaluate blacklist per folder per round
avg_score_per_folder = []
final_score_per_folder = []
for i in range(len(log_folders)):
    current_mal_devices = all_mal_devices[i]  # to compare blacklisted devices with
    # We want to know how many were correctly blacklisted at the end, but also how precise we were during learning.
    # Do an end-of-execution score (between 0 and 1, based on how many were correct).
    # Also do an average score for the entire learning process.
    folder_score = []
    for j in range(max_rounds[i]):
        # N.B. len(current_mal_devices) is an indicator of how many malicious devices we should mark as malicious.
        current_blacklisted = blacklisted_devices[i][j]
        # Compute the difference between current_mal_devices and current_blacklisted.
        not_blacklisted = [x for x in current_mal_devices if x not in current_blacklisted]
        wrongly_blacklisted = [y for y in current_blacklisted if y not in current_mal_devices and y != '']
        blacklist_score = (len(current_mal_devices) - (len(not_blacklisted) + len(wrongly_blacklisted))) / len(current_mal_devices)
        folder_score.append(blacklist_score)
        if j == max_rounds[i] - 1:
            final_score_per_folder.append(blacklist_score)
        print(current_mal_devices, not_blacklisted, wrongly_blacklisted)
    avg_score_per_folder.append(sum(folder_score)/len(folder_score))

print(avg_score_per_folder, final_score_per_folder)

print('Average blacklist score across all logs is: ' +
      ('%.2f' % (sum(avg_score_per_folder)/len(avg_score_per_folder))))
print('Average final blacklist score across all logs is: ' +
      ('%.2f' % (sum(final_score_per_folder)/len(final_score_per_folder))))

total_mal_devices = sum([len(i) for i in all_mal_devices])
avg_mal_devices = total_mal_devices / len(log_folders)

# Calculate proportion of time a malicious device was leader and the proportion of time they won if they were.
total_mal_leaders = sum([len(i) for i in indices_leader_per_folder])
prop_mal_leader = (total_mal_leaders / sum(max_rounds)) / avg_mal_devices
total_mal_won = sum([len(i) for i in indices_success_per_folder])
prop_mal_won = total_mal_won / total_mal_leaders

print('Malicious devices won ' + ('%.2f' % (prop_mal_won * 100)) + '% of the time they were assigned to be a leader.')
print('Malicious devices had a ' + ('%.2f' % (prop_mal_leader * 100)) + '% chance to be assigned leader.')

# with open(f'{fig_path}info/malicious_successes_nonIID_real.txt', 'a') as f:
#     f.write('Malicious devices won ' + ('%.2f' % (prop_mal_won * 100)) +
#             '% of the time they were assigned to be a leader. \n')
#     f.write('Malicious devices had a ' + ('%.2f' % (prop_mal_leader * 100)) + '% chance to be assigned leader.')
