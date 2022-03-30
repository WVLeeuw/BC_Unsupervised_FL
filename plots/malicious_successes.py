import matplotlib.pyplot as plt
import sys
import os
import block
from blockchain import Blockchain
import json

log_folders = ['03302022_180233']

# ToDo: rewrite this to be able to obtain the mal_indices per run. Have the y-axis correspond to the (different) runs.
# i.e. y_axis_labels = ['run_1', 'run_2'] etc. As VBFL did it.

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
indices_per_folder = []

for i in range(len(log_folders)):
    mal_devices = []
    malicious_devices_file = f'../logs/{log_folders[i]}/comm_1/malicious_devices_comm_1.txt'
    devices_file = open(malicious_devices_file, 'r')
    lines = devices_file.read().split('\n')
    for line in lines:
        if len(line) > 0:
            mal_devices.append(line.split()[0])
    all_mal_devices.append(mal_devices)

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
    indices_per_folder.append(indices_mal_winner)

print(all_mal_devices, indices_per_folder)
