import matplotlib.pyplot as plt
import sys
import os

log_folder = 'IID_real_3'

# Plot, for the provided log folder, the local model performance for ONE device (selected at random)
# vs. global model performance over n rounds.

fig_path = f'../logs/plots/'
dirs = os.listdir(f'../logs/{log_folder}')

max_round = 0
for folder in dirs:
    if 'comm_' in folder:
        if len(folder) > 7:
            max_round = 100
        elif len(folder) == 7:
            if int(folder[-2:]) > max_round:
                max_round = int(folder[-2:])
        elif len(folder) == 6:
            max_round = int(folder[-1])

# obtain no. devices during this execution.
args_used_file = f'../logs/{log_folder}/args_used.txt'
args_file = open(args_used_file, 'r')
lines_args = args_file.read().split('\n')
no_devices = 0
for line in lines_args:
    if 'num_devices' in line:
        no_devices = int(line[-2:])  # take the last two characters and cast to int

local_silhouettes = []
global_silhouettes = []

for j in range(1, max_round + 1):
    local_performance_file = f'../logs/{log_folder}/comm_{j}/silhouette_round_{j}.txt'
    round_info_file = f'../logs/{log_folder}/comm_{j}/round_{j}_info.txt'

    local_file = open(local_performance_file, 'r')
    info_file = open(round_info_file, 'r')

    lines_local = local_file.read().split('\n')
    round_scores = []
    for i in range(1, no_devices + 1):
        cur_device = f'device_{i}'
        for line in lines_local:
            if cur_device in line:
                round_scores.append(float(line.split(sep=':')[-1]))
    local_silhouettes.append(sum(round_scores)/len(round_scores))

    lines_info = info_file.read().split('\n')
    for line in lines_info:
        if 'Combining' in line:
            if 'silhouette' in line:
                global_silhouettes.append(float(line.split(sep=':')[-1]))

print(len(local_silhouettes), len(global_silhouettes))  # sanity check

colors = ['purple', 'orange']

fig = plt.figure(figsize=(8, 6))
ax = fig.subplots(1, 1)
ax.plot(range(1, max_round + 1), local_silhouettes, color=colors[0], label='Local silhouette (averaged)')
ax.plot(range(1, max_round + 1), global_silhouettes, color=colors[1], label='Global silhouette')
ax.set_title('Local and global model performance.')
ax.set_ylabel('Silhouette score')
ax.set_xlabel('Round number')
ax.set_ylim([-1, 1])
ax.legend()

filename = 'local_vs_global_performance_IID_real.png'
plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
plt.show()
