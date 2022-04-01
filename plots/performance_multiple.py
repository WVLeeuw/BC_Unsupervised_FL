import matplotlib.pyplot as plt
import sys
import os

log_folders = ['03312022_201510', '03312022_200231']

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

total_rounds = sum(max_rounds)  # used to get the eventual proportions

assert len(max_rounds) == len(log_folders), \
    'Could not find the number of communication rounds for (at least) one of the supplied runs.'

silhouette_avgs = []  # becomes list of lists, each inner list corresponding to a provided log folder.
davies_bouldin_scores = []  # idem, but for Davies-Bouldin index.

for i in range(len(log_folders)):
    silhouette_avg = []
    davies_bouldin = []
    for j in range(1, max_rounds[i] + 1):
        round_info_file = f'../logs/{log_folders[i]}/comm_{j}/round_{j}_info.txt'
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

colors = ['green', 'purple', 'orange', 'magenta']
print(len(silhouette_avgs), len(davies_bouldin_scores))  # sanity check.

fig = plt.figure(figsize=(8, 8))
ax1, ax2 = fig.subplots(2, 1, sharex=True)

for i in range(len(log_folders)):
    ax1.plot(range(1, max_rounds[i]+1), silhouette_avgs[i], color=colors[i])
ax1.set_title('Performance over time in terms of global model silhouette score.')
ax1.set_ylabel('silhouette score')
ax1.set_ylim([-1, 1])
ax1.legend(log_folders)

for i in range(len(log_folders)):
    ax2.plot(range(1, max_rounds[i]+1), davies_bouldin_scores[i], color=colors[i])
ax2.set_title('Performance over time in terms of global model Davies-Bouldin score.')
ax2.set_xlabel('round number')
ax2.set_ylabel('Davies-Bouldin score')
ax2.set_ylim([0, 2])
ax2.legend(log_folders)

plt.show()

for i in range(len(silhouette_avgs)):
    print(min(silhouette_avgs[i]), max(silhouette_avgs[i]))

for j in range(len(davies_bouldin_scores)):
    print(min(davies_bouldin_scores[j]), max(davies_bouldin_scores[j]))
