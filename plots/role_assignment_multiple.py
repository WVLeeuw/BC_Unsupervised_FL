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

B_none_count, M_none_count = 0, 0
B_data_owner_count, M_data_owner_count = 0, 0
B_committee_count, M_committee_count = 0, 0
B_leader_count, M_leader_count = 0, 0
for i in range(len(log_folders)):
    all_rounds = [f'comm_{i}' for i in range(1, max_rounds[i] + 1)]
    for round_iter in all_rounds:
        devices_assign_file = f'../logs/{log_folders[i]}/{round_iter}/role_assignment_{round_iter}.txt'
        file = open(devices_assign_file, 'r')
        lines_list = file.read().split('\n')
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
            else:  # must be a regular device.
                if 'None' in line:
                    B_none_count += 1
                if 'data owner' in line:
                    B_data_owner_count += 1
                if 'committee' in line:
                    B_committee_count += 1
                if 'leader' in line:
                    B_leader_count += 1

num_malicious = sum([M_none_count, M_data_owner_count, M_committee_count, M_leader_count]) / total_rounds
num_regular = sum([B_none_count, B_data_owner_count, B_committee_count, B_leader_count]) / total_rounds

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
malicious_props = [i/(num_malicious*total_rounds) for i in malicious_counts]
regular_props = [i/(num_regular*total_rounds) for i in regular_counts]

print(f'Malicious devices were assigned None, data owner, committee and leader with proportions {malicious_props} '
      f'respectively.')
print(f'Regular devices were assigned None, data owner, committee and leader with proportions {regular_props} '
      f'respectively.')

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
