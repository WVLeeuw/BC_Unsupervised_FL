import matplotlib.pyplot as plt
import sys
import os

log_folders = ['04042022_102601', '04042022_103057', '04042022_104125']

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

avg_time_spent = []
for i in range(1, max(max_rounds) + 1):
    times_this_round = []
    for j in range(len(log_folders)):
        round_info_file = f'../logs/{log_folders[j]}/comm_{i}/round_{i}_info.txt'
        if os.path.exists(round_info_file):
            file = open(round_info_file, 'r')
            lines_list = file.read().split('\n')
            for line in lines_list:
                if 'Time spent' in line:
                    times_this_round.append(float(line.split(sep=':')[-1].split()[0]))
        else:  # We have reached the final communication round for this log folder.
            break
    avg_time_spent.append(sum(times_this_round)/len(times_this_round))

print(f'Average total time per run is {sum(avg_time_spent)} seconds.')

fig = plt.figure(figsize=(8, 6))
ax1 = fig.subplots(1, 1)
ax1.plot(range(1, max(max_rounds) + 1), avg_time_spent)
ax1.set_title('Average time spent per learning round.')
ax1.set_ylabel('Time spent (s)')
ax1.set_xlabel('Round number')
ax1.set_ylim([0, 4])
plt.show()
