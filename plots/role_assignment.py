import matplotlib.pyplot as plt
import sys
import os

# log_folder = sys.argv[1]
log_folder = '03292022_122716'
dirs = os.listdir(f'../logs/{log_folder}')

max_round = 0
for folder in dirs:
    if 'comm_' in folder:
        if len(folder) == 6:
            max_round = int(folder[-1])
        elif len(folder) == 7:
            max_round = int(folder[-2:])
        else:
            max_round = 100

draw_comm_rounds = max_round  # could be less, could be more

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds + 1)]

B_none_count, M_none_count = 0, 0
B_data_owner_count, M_data_owner_count = 0, 0
B_committee_count, M_committee_count = 0, 0
B_leader_count, M_leader_count = 0, 0
for round_iter in all_rounds:
    devices_assign_file = f'../logs/{log_folder}/{round_iter}/role_assignment_{round_iter}.txt'
    file = open(devices_assign_file, 'r')
    lines_list = file.read().split('\n')
    # read for each device what role they have been assigned.
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
        else:
            if 'None' in line:
                B_none_count += 1
            if 'data owner' in line:
                B_data_owner_count += 1
            if 'committee' in line:
                B_committee_count += 1
            if 'leader' in line:
                B_leader_count += 1

num_malicious = sum([M_none_count, M_data_owner_count, M_committee_count, M_leader_count]) / draw_comm_rounds
num_regular = sum([B_none_count, B_data_owner_count, B_committee_count, B_leader_count]) / draw_comm_rounds

print(f"Malicious devices, of which there were {int(num_malicious)} in total, were assigned: "
      f"{M_none_count, M_data_owner_count, M_committee_count, M_leader_count}, "
      f"None, data owner, committee and leader respectively.")
print(f"Regular devices, of which there were {int(num_regular)} in total, were assigned: "
      f"{B_none_count, B_data_owner_count, B_committee_count, B_leader_count}, "
      f"None, data owner, committee and leader respectively.")

malicious_counts = [M_none_count, M_data_owner_count, M_committee_count, M_leader_count]
regular_counts = [B_none_count, B_data_owner_count, B_committee_count, B_leader_count]
labels = ['None', 'data owner', 'committee', 'leader']

# also obtain the proportions with which roles are assigned.
malicious_props = [i/(num_malicious*draw_comm_rounds) for i in malicious_counts]
regular_props = [i/(num_regular*draw_comm_rounds) for i in regular_counts]

fig = plt.figure(figsize=(8, 8))
ax1, ax2 = fig.subplots(2, 1, sharex=True)

ax1.bar(labels, malicious_props)
ax1.set_title('Role assignment for malicious nodes')
ax1.set_ylabel('proportion of rounds assigned')
ax1.set_ylim([0, 1])

ax2.bar(labels, regular_props)
ax2.set_title('Role assignment for regular devices')
ax2.set_xlabel('roles')
ax2.set_ylabel('proportion of rounds assigned')
ax2.set_ylim([0, 1])

plt.show()
