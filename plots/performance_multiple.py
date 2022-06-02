import matplotlib.pyplot as plt
import sys
import os

log_folders = [f'nonIID_dummy_{i}' for i in range(1, 21)]
fig_path = f'../logs/plots/'

max_rounds = [0 for i in range(len(log_folders))]
for i in range(len(log_folders)):
    cur_dir = os.listdir(f'../logs/{log_folders[i]}')
    for f in cur_dir:
        if 'comm_' in f:
            if len(f) > 7:
                max_rounds[i] = int(f[-3:])
            elif len(f) == 7:
                if int(f[-2:]) > max_rounds[i]:
                    max_rounds[i] = int(f[-2:])
            elif int(f[-1]) > max_rounds[i]:
                max_rounds[i] = int(f[-1])

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

colors = ['green', 'purple', 'orange', 'magenta', 'cyan']
print(len(silhouette_avgs), len(davies_bouldin_scores))  # sanity check.

fig = plt.figure(figsize=(8, 8))
ax1, ax2 = fig.subplots(2, 1, sharex=True)

for i in range(len(log_folders)):
    if len(silhouette_avgs[i]) > 0:
        ax1.plot(range(1, max_rounds[i]+1), silhouette_avgs[i])
ax1.set_title('Performance over time in terms of global model silhouette score.')
ax1.set_ylabel('silhouette score')
ax1.set_ylim([-1, 1])

for i in range(len(log_folders)):
    if len(davies_bouldin_scores[i]) > 0:
        ax2.plot(range(1, max_rounds[i]+1), davies_bouldin_scores[i])
ax2.set_title('Performance over time in terms of global model Davies-Bouldin score.')
ax2.set_xlabel('round number')
ax2.set_ylabel('Davies-Bouldin score')
ax2.set_ylim([0, 2])
fig.legend(log_folders, loc='right')

filename = 'performance_closely_real.png'
# plt.savefig(fname=os.path.join(fig_path, filename), dpi=600, bbox_inches='tight')
# plt.show()

final_silhouette_scores = [i[-1] for i in silhouette_avgs if len(i) > 0]
final_davies_bouldin_scores = [j[-1] for j in davies_bouldin_scores if len(j) > 0]

print(f"Average silhoutte score across all runs: {sum(final_silhouette_scores)/len(final_silhouette_scores)}.")
print(f"Average Davies-Bouldin score across all runs: "
      f"{sum(final_davies_bouldin_scores)/len(final_davies_bouldin_scores)}.")

# for i in range(len(silhouette_avgs)):
#     print(min(silhouette_avgs[i]), max(silhouette_avgs[i]))

# for j in range(len(davies_bouldin_scores)):
#     print(min(davies_bouldin_scores[j]), max(davies_bouldin_scores[j]))
