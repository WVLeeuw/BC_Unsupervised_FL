import matplotlib.pyplot as plt
import sys
import os

log_folder = '03302022_172717'

# Plot, for the provided log folder, the local model performance for ONE device (selected at random)
# vs. global model performance over n rounds.

dirs = os.listdir(f'../logs/{log_folder}')
device_idx = '12'  # number between 1 and n for n the number of devices.
cur_device = f'device_{device_idx}'

max_round = 0
for folder in dirs:
    if 'comm_' in folder:
        if len(folder) == 6:
            max_round = int(folder[-1])
        elif len(folder) == 7:
            max_round = int(folder[-2:])
        else:
            max_round = 100

local_silhouettes = []
global_silhouettes = []

for j in range(1, max_round + 1):
    local_performance_file = f'../logs/{log_folder}/comm_{j}/silhouette_round_{j}.txt'
    round_info_file = f'../logs/{log_folder}/comm_{j}/round_{j}_info.txt'

    local_file = open(local_performance_file, 'r')
    info_file = open(round_info_file, 'r')

    lines_local = local_file.read().split('\n')
    for line in lines_local:
        if cur_device in line:
            local_silhouettes.append(float(line.split(sep=':')[-1]))

    lines_info = info_file.read().split('\n')
    for line in lines_info:
        if 'Combining' in line:
            if 'silhouette' in line:
                global_silhouettes.append(float(line.split(sep=':')[-1]))

print(len(local_silhouettes), len(global_silhouettes))  # sanity check

colors = ['purple', 'orange']

fig = plt.figure(figsize=(8, 6))
ax = fig.subplots(1, 1)
ax.plot(range(1, max_round + 1), local_silhouettes, color=colors[0], label='Local silhouettes')
ax.plot(range(1, max_round + 1), global_silhouettes, color=colors[1], label='Global silhouettes')
ax.set_title('Local and global model performance.')
ax.set_ylabel('silhouette scores')
ax.set_xlabel('round numbers')
ax.set_ylim([-1, 1])
ax.legend()

plt.show()
