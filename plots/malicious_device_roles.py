import matplotlib.pyplot as plt
import sys

# log_folder = sys.argv[1]
log_folder = '03292022_123312'

draw_comm_rounds = 100  # could be less, could be more

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds+1)]

plt.figure(dpi=250, figsize=(6, 2))

none_count = 0
data_owner_count = 0
committee_count = 0
leader_count = 0
for round_iter in all_rounds:
    malicious_devices_assign_file = f'../logs/{log_folder}/{round_iter}/malicious_devices_{round_iter}.txt'
    file = open(malicious_devices_assign_file, 'r')
    lines_list = file.read().split('\n')
    # read for each device what role they have been assigned.
    for line in lines_list:
        if 'data owner' in line:
            data_owner_count += 1
        if 'committee' in line:
            committee_count += 1
        if 'leader' in line:
            leader_count += 1
        if 'None' in line:
            none_count += 1
print(none_count, data_owner_count, committee_count, leader_count)
