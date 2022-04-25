import json
import math
import argparse
import copy
import math
import os
import pickle
import random
import shutil
import sys
import time
from datetime import datetime
from sys import getsizeof

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

import KMeans
from blockchain import Blockchain
import block
from device import DevicesInNetwork
from utils import data_utils

date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_folder_path = f"logs/{date_time}"
bc_folder = f"blockchains/{date_time}"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='BCFL_kmeans_Simulation')

# debug attributes
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')

# FL attributes
parser.add_argument('-data', '--dataset', type=str, default='', help='dataset to be used')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communication rounds')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='number of devices in the simulation')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help='number of malicious devices')
parser.add_argument('-iid', '--IID', type=int, default=1, help='whether to allocate data in iid setting')
parser.add_argument('-gc', '--num_global_centroids', type=int, default=3, help='number of centroids in the globally '
                                                                               'trained model')
parser.add_argument('-le', '--num_local_epochs', type=int, default=1, help='number of epochs to perform for the '
                                                                           'acquisition of local update')
parser.add_argument('-fa', '--fed_avg', type=int, default=1, help='whether to use Federated Averaging')
parser.add_argument('-eps', '--epsilon', type=float, default=0.01,
                    help='threshold for the difference between the location of newly computed centroids and previous '
                         'global centroids s.t. when that difference is less than epsilon, the learning process ends.')
parser.add_argument('-gul', '--global_update_lag', type=float, default=0.0,
                    help='parameter representing the lag of global model updates. Possible values between 0.0 and 1.0,'
                         'where a higher value represents more lag which in turn translates to new model updates '
                         'having less effect on the model update.')
parser.add_argument('-ninit', '--num_reinitialize', type=int, default=5,
                    help='parameter representing how often the global learning process may reset if it deems the '
                         'initial centroid positions to be poor (and thus decides that proper convergence is '
                         'unlikely).')

# Additional BC attributes (to make entire process more efficient)
parser.add_argument('-cmut', '--committee_member_update_wait_time', type=float, default=0.0,
                    help="time window during which committee members wait for local updates to be sent. Wait time of "
                         "0.0 is associated with no time limit.")
parser.add_argument('-cmbt', '--committee_member_block_wait_time', type=float, default=0.0,
                    help="time window during which committee member wait for block proposals to be sent. Wait time of "
                         "0.0 is associated with no time limit.")
parser.add_argument('-cmh', '--committee_member_threshold', type=float, default=0.0,
                    help="threshold value for the difference in performance to determine whether to consider an update")
parser.add_argument('-lwt', '--leader_wait_time', type=float, default=0.0,
                    help="time window during which leaders wait for committee members to send their resulting "
                         "aggregate after they obtained the local updates from data owners. Wait time of 0.0 is "
                         "associated with no time limit.")
parser.add_argument('-rfc', '--resume_from_chain', type=str, default=None,
                    help='resume the global learning process from an existing blockchain from the path of a saved '
                         'blockchain. Only provide the date.')
parser.add_argument('-rs', '--reputation_system', type=int, default=1,
                    help='whether to assign roles based on the reputation system.')
parser.add_argument('-cth', '--contribution_threshold', type=float, default=-.2,
                    help="threshold value for the contribution. If a device's contribution lower than the provided "
                         "value, the device will be excluded from being a data owner entirely.")
parser.add_argument('-rr', '--reputation_ratio', type=float, default=3.0,
                    help="ratio of negative interactions to positive interactions in which case a device is excluded "
                         "from being a committee member or leader.")

# distributed system attributes
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds of a device being and '
                                                                                'staying online')
parser.add_argument('-els', '--equal_link_speed', type=int, default=1,
                    help='used to simulate transmission delay. If set to 1, every device has equal link speed ('
                         'bytes/sec). If set to 0, link speed is determined randomly.')
parser.add_argument('-dts', '--data_transmission_speed', type=float, default=70000.0,
                    help="volume of data that is transmitted per second when -els == 1.")
parser.add_argument('-ecp', '--equal_computation_power', type=int, default=1,
                    help='used to simulation computation power. If set to 1, every device has equal computation '
                         'power. If set to 0, computation power is determined randomly.')
parser.add_argument('-contr', '--contribution_lag_param', type=float, default=.5,
                    help="parameter used when computing a device's contribution during a learning round. It is "
                         "referred to as lag because the parameter determines how heavily contribution from past "
                         "rounds weighs into the computation of the most recent round's contribution of the device.")
parser.add_argument('-cc', '--closely_connected', type=str, default='closely',
                    help='whether to have each data owner be connected to all committee members, have the '
                         'connection be one to one or have a handful of committee members see each update.'
                         'Possible arguments are "loosely", "closely" and "somewhat", the latter meaning'
                         'that a handful of committee members see each update.')

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*',
                    help='number of devices assigned to the roles of data owner, committee member and leader '
                         'respectively')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='whether to check signatures, used to save '
                                                                          'time or to assume trust')
parser.add_argument('-aio', '--all_in_one_network', type=int, default=1,
                    help='whether to have all devices be aware of and connected to each other device in the network')

if __name__ == '__main__':

    # get arguments
    args = parser.parse_args()
    args = args.__dict__

    # 0. create directory for log files
    if not os.path.isdir(log_folder_path):
        os.mkdir(log_folder_path)

    # 1. save the arguments that were used
    with open(f'{log_folder_path}/args_used.txt', 'w') as f:
        f.write("Command line arguments used -\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nAll arguments used -\n")
        for arg_name, arg in args.items():
            f.write(f'\n--{arg_name} {arg}')

    # 2. create folder to save models in, if it does not already exist
    if not os.path.isdir(bc_folder):
        os.mkdir(bc_folder)

    # 3. get number of devices per role required in the network
    roles_requirement = args['hard_assign'].split(',')
    # determine roles to assign
    data_owners_needed = int(roles_requirement[0])
    committee_members_needed = int(roles_requirement[1])
    leaders_needed = int(roles_requirement[2])

    # 4. check eligibility of arguments
    num_devices = args['num_devices']
    num_malicious = args['num_malicious']

    if num_devices < data_owners_needed + committee_members_needed + leaders_needed:
        sys.exit("ERROR: Roles assigned to devices exceed the maximum number of devices.")

    if num_devices < 3:
        sys.exit("ERROR: There are not enough devices in the network to execute. \n There need to be at least one "
                 "data owner, one committee member and one committee leader.")

    if num_malicious:
        if num_malicious > num_devices:
            sys.exit("ERROR: The number of malicious devices cannot exceed the total number of devices.")
        else:
            print(f'Malicious nodes and total devices set to {num_malicious}:{num_devices}')

    # 5. Create devices in the network.
    devices_in_network = DevicesInNetwork(is_iid=args['IID'], num_devices=num_devices, num_malicious=num_malicious,
                                          local_epochs=args['num_local_epochs'],
                                          network_stability=args['network_stability'],
                                          committee_update_wait_time=args['committee_member_update_wait_time'],
                                          committee_block_wait_time=args['committee_member_block_wait_time'],
                                          committee_threshold=args['committee_member_threshold'],
                                          contribution_lag=args['contribution_lag_param'],
                                          fed_avg=args['fed_avg'], leader_wait_time=args['leader_wait_time'],
                                          global_update_lag=args['global_update_lag'],
                                          equal_link_speed=args['equal_link_speed'],
                                          data_transmission_speed=args['data_transmission_speed'],
                                          equal_computation_power=args['equal_computation_power'],
                                          check_signature=args['check_signature'],
                                          stop_condition=args['epsilon'], dataset=args['dataset'])
    device_list = list(devices_in_network.devices_set.values())

    # Extract the bounds on the data which was used for the creation of the devices.
    # Also obtain all idxs for the initialization of reputation and contribution in genesis block.
    datasets = []
    idxs = []
    for device in device_list:
        datasets.append(device.dataset)
        idxs.append(device.return_idx())

    min_vals, max_vals = data_utils.obtain_bounds_multiple(np.asarray(datasets))
    bounds = []
    for i in range(len(min_vals)):  # N.B. len(min_vals) should be equal to n_dims every single time.
        bounds.append([min_vals[i], max_vals[i]])
    n_dims, n_clusters = len(bounds), args['num_global_centroids']

    # Finally, produce the genesis block with initialized parameters.
    data = dict()
    init_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims, n_clusters)
    centroids = init_centroids
    bc = Blockchain()
    bc.create_genesis_block(device_idxs=idxs, centroids=centroids)
    track_g_centroids = [init_centroids]

    # 6. register devices and initialize global parameters including genesis block.
    for device in device_list:
        if args['resume_from_chain']:
            bc_save_path = f"blockchains/{args['resume_from_chain']}/"
            blocks = []
            for f in os.listdir(bc_save_path):
                if f.endswith('.json'):
                    cur_block = json.load(open(bc_save_path + '/' + f))
                    blocks.append(block.fromJSON(cur_block))

            blocks = sorted(blocks, key=lambda x: x.index, reverse=False)
            chain = Blockchain()

            for b in blocks:
                chain.append_block(b)

            if chain.is_chain_valid():  # feed the blockchain to the devices if it is valid.
                device.blockchain = copy.copy(chain)
            else:  # start from chain with only genesis block.
                device.blockchain = copy.copy(bc)
        else:
            # feed the created blockchain with genesis block to each device.
            device.blockchain = copy.copy(bc)

        device.initialize_kmeans_model(n_dims=n_dims, n_clusters=device.elbow_method())
        # simulates peer registration, connects to some or all devices depending on 'all_in_one_network'.
        device.set_devices_dict_and_aio(devices_in_network.devices_set, args['all_in_one_network'])
        device.register_in_network()
    if args['verbose']:
        print(str(device_list[-1].obtain_latest_block()))
    # remove the device if it is in its own peer list
    for device in device_list:
        device.remove_peers(device)

    # 7. run elbow method to make a sensible choice for global k.
    k_choices = []
    for device in device_list:
        k_choices.append(device.elbow_method())
    print("Average choice of k found: " + str(math.ceil(sum(k_choices) / len(k_choices))))
    print(math.ceil(sum(k_choices) / len(k_choices)) == args['num_global_centroids'])
    # check if avg equals supplied parameter for global centroids.

    # 8. initialize blacklists for devices
    blacklist_contr = []
    blacklist_rep = []

    # BCFL-KMeans starts here
    time_taken_per_round = []
    est_time_taken_parallel_per_round = []
    total_comm_rounds = 0
    comm_round = 0
    num_rounds_no_winner = 0
    num_reinitialized = 0
    while total_comm_rounds < args['num_comm']:
        # create log folder for communication round
        log_folder_path_comm_round = f"{log_folder_path}/comm_{comm_round + 1}"
        if os.path.exists(log_folder_path_comm_round):
            print(f"Deleting {log_folder_path_comm_round} and creating a new one.")
            shutil.rmtree(log_folder_path_comm_round)
        os.mkdir(log_folder_path_comm_round)
        print(f"\nCommunication round {comm_round + 1}.")

        # Log initial centers
        if comm_round == 0:
            with open(f"{log_folder_path_comm_round}/initial_centers.data", 'wb') as file:
                pickle.dump(init_centroids, file)

        comm_round_start_time = time.time()  # to keep track how long communication rounds take.
        parallel_time_estimate = 0  # to keep track of the time it would take if every device ran in parallel
        total_comm_rounds += 1  # to keep track of the total nr of communication rounds.

        # i. assign roles to devices dependent on contribution and reputation
        data_owners_to_assign = data_owners_needed
        committee_members_to_assign = committee_members_needed
        leaders_to_assign = leaders_needed

        data_owners_this_round = []
        committee_members_this_round = []
        leaders_this_round = []

        # For each device, draw a sample from its beta distribution (dependent on its reputation),
        # moreover, use the contribution value to determine whether they can be a data owner during this round.
        # N.B. we assume that this sampling is done through smart contract at start of round.
        if args['reputation_system']:
            latest_block_data = device_list[-1].obtain_latest_block().get_data()
            pos_rep = latest_block_data['pos_reputation']
            neg_rep = latest_block_data['neg_reputation']
            contr_vals = latest_block_data['contribution']
            contr_bounds = [contr_vals[min(contr_vals.keys(), key=(lambda k: contr_vals[k]))],
                            contr_vals[max(contr_vals.keys(), key=(lambda k: contr_vals[k]))]]
            print(f"Bounds of the contribution values this round are: {contr_bounds}")
            eligible_comm_members = {}
            eligible_leaders = {}
            chosen_catch_up = []  # 10% probability that a device having (1, 1) reputation is chosen.
            for device in device_list:
                if device.return_idx() in pos_rep and device.return_idx() in neg_rep:
                    pos_count, neg_count = pos_rep[device.return_idx()], neg_rep[device.return_idx()]
                else:  # initialize both to be 1 (as a new device must have just joined).
                    pos_count, neg_count = 1, 1
                # pos_count == neg_count == 1 corresponds to new devices (in theory).
                if pos_count == neg_count == 1:
                    # assign committee role with 5% probability.
                    if random.random() > .95 and len(chosen_catch_up) <= committee_members_to_assign // 3:
                        chosen_catch_up.append(device.return_idx())

                # Check whether the device should be blacklisted based on their negative and positive interactions.
                if neg_count / pos_count >= args['reputation_ratio']:
                    blacklist_rep.append(device.return_idx())

                # N.B. if a device's contribution is poor, they would not do well to validate other updates either.
                if device.return_idx() not in blacklist_rep + blacklist_contr:
                    # check whether the device was a committee member in the previous round, not eligible otherwise.
                    if device.return_role() != 'committee':
                        eligible_comm_members[device.return_idx()] = np.random.beta(pos_count, neg_count)
                    # idem for leaders.
                    if device.return_role() != 'leader':
                        eligible_leaders[device.return_idx()] = np.random.beta(pos_count, neg_count)

            print(f"{len(chosen_catch_up)} device(s) were chosen to catch up!")
            sorted_eligible_comm = {k: v for k, v in sorted(eligible_comm_members.items(), key=lambda item: item[1],
                                                            reverse=True)}
            eligible_comm_filtered = dict(list(sorted_eligible_comm.items())[:committee_members_to_assign])
            # Extract the keys, remove keys if there were devices selected to 'catch up'.
            eligible_comm_keys = list(eligible_comm_filtered.keys())
            sorted_eligible_leaders = {k: v for k, v in sorted(eligible_leaders.items(), key=lambda item: item[1],
                                                               reverse=True)}
            eligible_leaders_filtered = dict(list(sorted_eligible_leaders.items())[:leaders_to_assign])
            eligible_leaders_keys = list(eligible_leaders_filtered.keys())

            random.shuffle(eligible_comm_keys)
            random.shuffle(eligible_leaders_keys)

            # reset the device's role from previous round, assign previously drawn devices (from sampling) first.
            for device in device_list:
                device.reset_role()
                if device.return_idx() in chosen_catch_up and committee_members_to_assign:
                    device.assign_committee_role()
                    committee_members_to_assign -= 1

                elif device.return_idx() in eligible_leaders_keys:
                    if leaders_to_assign:
                        device.assign_leader_role()
                        leaders_to_assign -= 1
                elif device.return_idx() in eligible_comm_keys:
                    if committee_members_to_assign:
                        device.assign_committee_role()
                        committee_members_to_assign -= 1

                # Done assigning committee members and leaders, adding them to their respective lists.
                if device.return_role() == "leader":
                    leaders_this_round.append(device)
                if device.return_role() == "committee":
                    committee_members_this_round.append(device)

            still_unassigned = [device for device in device_list if not device.return_role()]
            if contr_bounds[0] < 0.0:
                for device in still_unassigned:
                    if contr_vals[device.return_idx()] < args['contribution_threshold']:
                        blacklist_contr.append(device.return_idx())
                    elif data_owners_to_assign and device.return_idx() not in blacklist_contr:
                        contr_range = abs(contr_bounds[1] - contr_bounds[0])
                        # Exclude the bottom 25% of contributors if there are negative contribution values.
                        if device.return_idx() in contr_vals:
                            if contr_vals[device.return_idx()] > contr_bounds[0] + .25 * contr_range:
                                device.assign_data_role()
                                data_owners_to_assign -= 1
                            # And have each device has a 10% chance otherwise to be picked to 'catch up'.
                            elif random.random() > 0.9:
                                print(f"{device.return_idx()} has been chosen to provide an update even though "
                                      f"their contribution and/or reputation is poor.")
                                device.assign_data_role()
                                data_owners_to_assign -= 1
                        elif random.random() > 0.9:  # reached if a new device has just joined.
                            device.assign_data_role()
                            data_owners_to_assign -= 1
                    # Then append the devices to the data owner list.
                    if device.return_role() == 'data owner':
                        data_owners_this_round.append(device)
            else:  # every device must have positive contribution -- no filtering required.
                for device in still_unassigned:
                    if data_owners_to_assign:
                        device.assign_data_role()
                        data_owners_to_assign -= 1
                        data_owners_this_round.append(device)

        else:  # do random assignment, with caveat that devices cannot be leaders/committee in consecutive rounds.
            # check if the devices are eligible to be either leader or committee
            eligible_leaders = [device for device in device_list if device.return_role() != 'leader']
            eligible_committee = [device for device in device_list if device.return_role() != 'committee']
            # shuffle the device list to simulate random assignments
            random.shuffle(device_list)
            for device in device_list:
                if leaders_to_assign and device in eligible_leaders:
                    device.assign_leader_role()
                    leaders_to_assign -= 1
                elif committee_members_to_assign and device in eligible_committee:
                    device.assign_committee_role()
                    committee_members_to_assign -= 1
                elif data_owners_to_assign:
                    device.assign_data_role()
                    data_owners_to_assign -= 1

                # Done assigning devices, adding them to their respective lists.
                if device.return_role() == "leader":
                    leaders_this_round.append(device)
                if device.return_role() == "committee":
                    committee_members_this_round.append(device)
                if device.return_role() == "data owner":
                    data_owners_this_round.append(device)

        # finally, check whether the devices are online.
        for device in device_list:
            device.online_switcher()

        # log roles assigned for the devices, along with their reputation and contribution values.
        for device in device_list:
            # N.B. We can also keep track of how often a device is selected for a (specific) role while being malicious.
            role_assigned = device.return_role()
            latest_block_data = device.obtain_latest_block().get_data()
            if device.return_idx() in latest_block_data['contribution'] and \
                    device.return_idx() in latest_block_data['pos_reputation'] and \
                    device.return_idx() in latest_block_data['neg_reputation']:
                contribution_val = latest_block_data['contribution'][device.return_idx()]
                reputation_val = (latest_block_data['pos_reputation'][device.return_idx()],
                                  latest_block_data['neg_reputation'][device.return_idx()])
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                with open(f"{log_folder_path_comm_round}/role_assignment_comm_{comm_round + 1}.txt", 'a') as file:
                    file.write(f"{device.return_idx()} {is_malicious_node} assigned {device.return_role()} having "
                               f"contribution {contribution_val} and reputation {reputation_val}. \n")
                with open(f"{log_folder_path_comm_round}/malicious_devices_comm_{comm_round + 1}.txt", 'a') as file:
                    if device.return_is_malicious():
                        file.write(f"{device.return_idx()} assigned {device.return_role()}. \n")

        if args['verbose']:
            print(f"Number of leaders this round: {len(leaders_this_round)} \n"
                  f"Number of committee members this round: {len(committee_members_this_round)} \n"
                  f"Number of data owners this round: {len(data_owners_this_round)}")

        # Reset variables for this communication round.
        for data_owner in data_owners_this_round:
            data_owner.reset_vars_data_owner()
        for comm_member in committee_members_this_round:
            comm_member.reset_vars_committee_member()
        for leader in leaders_this_round:
            leader.reset_vars_leader()

        role_assignment_time = time.time() - comm_round_start_time
        print(f"Assigning roles took {role_assignment_time} seconds.")

        # ii. obtain most recent block and perform local learning step and share result with associated committee member
        for device in data_owners_this_round:
            if device.return_is_malicious():
                device.malicious_local_update()
            else:
                device.local_update()

            # Send the result to a committee member in the device's peer list.
            # Depending on the closeness of connections, put the data owner either in every committee member's
            # associated set, or only put them in a single committee member's associated set.
            if args['closely_connected'] == 'closely':
                for peer in device.return_peers():
                    if peer in committee_members_this_round:
                        peer.add_device_to_associated_set(device)
            else:
                eligible_comm_members = []
                for peer in device.return_peers():
                    if peer in committee_members_this_round:
                        eligible_comm_members.append(peer)
                random.shuffle(eligible_comm_members)
                # Associate the device with only one committee member.
                if args['closely_connected'] == 'loosely':
                    eligible_comm_members[0].add_device_to_associated_set(device)
                else:  # have at most half of the committee members be associated with the data owner
                    eligible_comm_members_filtered = eligible_comm_members[:len(committee_members_this_round) // 2]
                    for comm_member in eligible_comm_members_filtered:
                        comm_member.add_device_to_associated_set(device)

        # iii. committee members validate retrieved updates and aggregate viable results
        max_local_update_time = 0  # including transmission
        max_validation_time = 0
        for comm_member in committee_members_this_round:
            global_centroids = comm_member.retrieve_global_centroids()
            if args['committee_member_update_wait_time']:
                print(f"Committee local update wait time is specified as {args['committee_member_update_wait_time']} "
                      f"seconds. Allowing each data owner to perform local training until time limit.")
                for data_owner in comm_member.return_online_associated_data_owners():
                    total_time_tracker = 0
                    data_owner_link_speed = data_owner.return_link_speed()
                    lower_link_speed = comm_member.return_link_speed() if \
                        comm_member.return_link_speed() < data_owner_link_speed else data_owner_link_speed
                    if data_owner.online_switcher():
                        local_update_spent_time = data_owner.return_local_update_time()
                        local_update, nr_records, data_owner_idx = data_owner.return_local_update()
                        local_update_size = getsizeof([local_update, nr_records, data_owner_idx])
                        transmission_delay = local_update_size / lower_link_speed
                        local_update_total_time = local_update_spent_time + transmission_delay

                        if local_update_total_time > max_local_update_time:
                            max_local_update_time = local_update_total_time
                        # check whether the time taken was less than the allowed time for local updating.
                        if local_update_total_time < comm_member.return_update_wait_time():
                            if comm_member.online_switcher():
                                comm_member.obtain_local_update(local_update, nr_records, data_owner_idx)
                                with open(f"{log_folder_path_comm_round}/silhouette_diffs_{comm_round + 1}.txt",
                                          'a') as file:
                                    data_owner_silhouette = data_owner.validate_update(local_update)
                                    data_owner_is_malicious_node = "M" if data_owner.return_is_malicious() else "B"
                                    comm_member_silhouette = comm_member.validate_update(local_update)
                                    file.write(f"{comm_member_silhouette - data_owner_silhouette}, data owner "
                                               f"{data_owner.return_idx()} {data_owner_is_malicious_node} local "
                                               f"silhouette score was {data_owner_silhouette} while the committee "
                                               f"member {comm_member.return_idx()} obtained a silhouette score of "
                                               f"{comm_member_silhouette} \n")
                    else:
                        print(f"Data owner {data_owner.return_idx()} is unable to perform local update due to being "
                              f"offline.")
            else:
                for data_owner in comm_member.return_online_associated_data_owners():
                    data_owner_link_speed = data_owner.return_link_speed()
                    lower_link_speed = comm_member.return_link_speed() if \
                        comm_member.return_link_speed() < data_owner_link_speed else data_owner_link_speed
                    if data_owner.online_switcher():
                        local_update_spent_time = data_owner.return_local_update_time()
                        local_update, nr_records, data_owner_idx = data_owner.return_local_update()
                        local_update_size = getsizeof([local_update, nr_records, data_owner_idx])
                        transmission_delay = local_update_size / lower_link_speed
                        local_update_total_time = local_update_spent_time + transmission_delay

                        if local_update_total_time > max_local_update_time:
                            max_local_update_time = local_update_total_time

                        # finally, obtain the local update.
                        if comm_member.online_switcher():
                            comm_member.obtain_local_update(local_update, nr_records, data_owner_idx)
                            with open(f"{log_folder_path_comm_round}/silhouette_diffs_{comm_round + 1}.txt",
                                      'a') as file:
                                data_owner_silhouette = data_owner.validate_update(local_update)
                                data_owner_is_malicious_node = "M" if data_owner.return_is_malicious() else "B"
                                comm_member_silhouette = comm_member.validate_update(local_update)
                                file.write(f"{comm_member_silhouette - data_owner_silhouette}, data owner "
                                           f"{data_owner.return_idx()} {data_owner_is_malicious_node} local "
                                           f"silhouette score was {data_owner_silhouette} while the committee member "
                                           f"{comm_member.return_idx()} obtained a silhouette score of "
                                           f"{comm_member_silhouette} \n")
                    else:
                        print(f"Data owner {data_owner.return_idx()} is unable to perform local update due to being "
                              f"offline.")

            # validate local updates and aggregate usable local updates
            if comm_member.online_switcher():
                if comm_member.return_is_malicious():
                    aggr_centroids = comm_member.malicious_aggr_updates()
                if comm_member.return_fed_avg():
                    aggr_centroids = comm_member.aggr_fed_avg()
                else:
                    updates_per_centroid = comm_member.match_local_with_global_centroids()
                    aggr_centroids = comm_member.aggr_updates(updates_per_centroid)

                validation_time = comm_member.return_validation_time()
                if validation_time > max_validation_time:
                    max_validation_time = validation_time

                if args['verbose']:
                    print(aggr_centroids)
                    print(str(comm_member.validate_update(aggr_centroids)) +
                          " compared to previous global model performance of " +
                          str(comm_member.validate_update(comm_member.retrieve_global_centroids())))

        # iv. committee members send updated centroids to every leader
        max_aggr_time = 0
        for leader in leaders_this_round:
            if args['leader_wait_time']:
                for comm_member in leader.return_online_committee_members():
                    comm_member_link_speed = comm_member.return_link_speed()
                    lower_link_speed = leader.return_link_speed() if \
                        leader.return_link_speed() < comm_member_link_speed else comm_member_link_speed
                    if comm_member.online_switcher():
                        aggregation_spent_time = comm_member.return_aggregation_time()
                        centroids, feedback, idx = comm_member.return_aggregate_and_feedback()
                        aggregate_and_feedback_size = getsizeof([centroids, feedback, idx])
                        transmission_delay = aggregate_and_feedback_size / lower_link_speed
                        aggr_and_feedback_total_time = aggregation_spent_time + transmission_delay

                        if aggr_and_feedback_total_time > max_aggr_time:
                            max_aggr_time = aggr_and_feedback_total_time
                        # finally, obtain the aggregate and associated feedback.
                        if aggr_and_feedback_total_time < leader.return_aggr_wait_time():
                            if leader.online_switcher():
                                leader.obtain_aggr_and_feedback(centroids, feedback, idx)
            else:
                for comm_member in leader.return_online_committee_members():
                    comm_member_link_speed = comm_member.return_link_speed()
                    lower_link_speed = leader.return_link_speed() if \
                        leader.return_link_speed() < comm_member_link_speed else comm_member_link_speed
                    if comm_member.online_switcher():
                        aggregation_spent_time = comm_member.return_aggregation_time()
                        centroids, feedback, idx = comm_member.return_aggregate_and_feedback()
                        aggregate_and_feedback_size = getsizeof([centroids, feedback, idx])
                        transmission_delay = aggregate_and_feedback_size / lower_link_speed
                        aggr_and_feedback_total_time = aggregation_spent_time + transmission_delay

                        if aggr_and_feedback_total_time > max_aggr_time:
                            max_aggr_time = aggr_and_feedback_total_time
                        # finally, obtain the aggregate and associated feedback.
                        if leader.online_switcher():
                            leader.obtain_aggr_and_feedback(centroids, feedback, idx)

            # v. leaders build candidate blocks using the obtained centroids and send it to committee members
            # for approval.
            if leader.return_is_malicious():
                proposed_g_centroids = leader.malicious_compute_update()
                block = leader.malicious_build_block(proposed_g_centroids)
            else:
                proposed_g_centroids = leader.compute_update()
                block = leader.build_block(proposed_g_centroids)
            if args['verbose']:
                print(str(block), '\n')

            # log proposed block
            with open(f"{log_folder_path_comm_round}/proposed_blocks_round_{comm_round + 1}.txt", 'a') as file:
                is_malicious_node = "M" if leader.return_is_malicious() else "B"
                file.write(f"{leader.return_idx()} {is_malicious_node} "
                           f"proposed block: \n {leader.return_proposed_block()}. \n\n")

        # Determine the arrival order of proposed blocks.
        block_arrival_queue = {}
        max_proposal_time = 0  # including transmission
        for leader in leaders_this_round:
            if args['committee_member_block_wait_time']:
                for comm_member in leader.return_online_committee_members():
                    comm_member_link_speed = comm_member.return_link_speed()
                    lower_link_speed = leader.return_link_speed() if \
                        leader.return_link_speed() < comm_member_link_speed else comm_member_link_speed
                    if leader.online_switcher():
                        block, block_time = leader.broadcast_block()
                        block_size = getsizeof(block)
                        transmission_delay = block_size / lower_link_speed
                        total_block_time = block_time + transmission_delay

                        if total_block_time > max_proposal_time:
                            max_proposal_time = total_block_time
                        # finally, send the proposed block.
                        if total_block_time < comm_member.return_block_wait_time():
                            if comm_member.online_switcher():
                                comm_member.add_to_candidate_blocks(block, total_block_time)
                                if total_block_time not in block_arrival_queue:
                                    block_arrival_queue[total_block_time] = [block, [comm_member.return_idx()]]
                                else:
                                    block_arrival_queue[total_block_time][-1].append(comm_member.return_idx())
            else:
                for comm_member in leader.return_online_committee_members():
                    comm_member_link_speed = comm_member.return_link_speed()
                    lower_link_speed = leader.return_link_speed() if \
                        leader.return_link_speed() < comm_member_link_speed else comm_member_link_speed
                    if leader.online_switcher():
                        block, block_time = leader.broadcast_block()
                        block_size = getsizeof(block)
                        transmission_delay = block_size / lower_link_speed
                        total_block_time = block_time + transmission_delay

                        if total_block_time > max_proposal_time:
                            max_proposal_time = total_block_time
                        # finally, send the proposed block.
                        if comm_member.online_switcher():
                            comm_member.add_to_candidate_blocks(block, total_block_time)
                            if total_block_time not in block_arrival_queue:
                                block_arrival_queue[total_block_time] = [block, [comm_member.return_idx()]]
                            else:
                                block_arrival_queue[total_block_time][-1].append(comm_member.return_idx())

        # in-between step: determine order in which to deal with candidate blocks.
        block_arrival_queue = dict(sorted(block_arrival_queue.items()))

        # vi. committee members vote on candidate blocks by sending their vote to all committee members (signed)
        winning_block = False
        winner = None
        votes = {}
        # Voting stage!
        for arrival_time, block_comm_pairs in block_arrival_queue.items():
            # block_comm_pairs is a pair, containing the block and the idxs of the committee members that retrieved
            # it at that time.
            candidate_block = block_comm_pairs[0]
            for comm_member_idx in block_comm_pairs[-1]:
                for comm_member in committee_members_this_round:
                    if comm_member.return_idx() == comm_member_idx:
                        if comm_member.online_switcher():
                            if comm_member.verify_signature(candidate_block):
                                print(f"Candidate block from {candidate_block.get_produced_by()} "
                                      f"has been verified by {comm_member.return_idx()}")
                                centroids_to_check = candidate_block.get_data()['centroids']
                                print("Using the newly proposed global centroids from the candidate block, "
                                      "an average ""silhouette score of " +
                                      str(comm_member.validate_update(centroids_to_check)) + " was found.")
                                vote = comm_member.approve_block(candidate_block)
                                time_to_vote = comm_member.return_block_validation_time()
                                if vote:
                                    leader_idx = candidate_block.get_produced_by()
                                    votes[time_to_vote] = [vote, comm_member.return_idx(), leader_idx]
                            else:
                                print("Candidate block could not be verified. \n"
                                      "Block originates from device having idx: " +
                                      str(candidate_block.get_produced_by()))

        # in-between step: determine order in which the votes would finally arrive.
        final_vote_order = {}
        for vote_time, vote in votes.items():
            vote_size = getsizeof(vote[0])
            comm_member_idx = vote[1]
            leader_idx = vote[2]
            for comm_member in committee_members_this_round:
                if comm_member.return_idx() == comm_member_idx:
                    comm_member_link_speed = comm_member.return_link_speed()
                    for leader in leaders_this_round:
                        if leader.return_idx() == leader_idx:
                            leader_link_speed = leader.return_link_speed()
                            lower_link_speed = leader_link_speed if leader_link_speed < comm_member_link_speed \
                                else comm_member_link_speed
                            transmission_delay = vote_size / lower_link_speed
                            final_vote_order[transmission_delay + vote_time] = vote
        final_vote_order = dict(sorted(final_vote_order.items()))
        # maximum time can be retrieved by taking the last item of the final_vote_order keys.
        latest_vote_cast_time = list(final_vote_order.keys())[-1] if len(list(final_vote_order.keys())) > 0 else 0
        print(f"Latest time that a vote was cast was after {latest_vote_cast_time} seconds.")

        # Counting stage!
        # A vote is a pair [0] contains the message, [1] contains the committee idx, [2] contains the leader idx.
        winning_block = False
        winner = None
        block_completion_time = 0
        for vote_time, vote in final_vote_order.items():
            msg = vote[0]
            comm_member_idx = vote[1]
            leader_idx = vote[2]
            if winning_block:  # check whether we already found a winner.
                break
            for comm_member in committee_members_this_round:
                if comm_member.return_idx() == comm_member_idx:
                    comm_member.send_vote(msg, leader_idx)
                    print(f"{comm_member.return_idx()} voted for {leader_idx}'s block")
                    # Find the leader that just obtained the vote.
                    for leader in leaders_this_round:
                        if leader.return_idx() == leader_idx:
                            if len(leader.return_received_votes()) > len(committee_members_this_round) // 2:
                                winning_block = True
                                winner = leader
                                winner.serialize_votes()
                                block_completion_time = vote_time
                                print(f"{leader_idx} was the first to receive a majority vote for their block.")
                                break

        # vii. leader that obtained a majority vote append their block to their chain and propagate it to all
        # committee members. Afterward, committee members request their peers to download the winning block.
        total_broadcast_delay = 0  # defining broadcast and propagation delays here because it is not reachable
        total_propagation_delay = 0  # by parallel_time_estimate (computation) otherwise.
        re_init_event = False
        if winning_block:
            # check whether there is a centroid which was not updated. We do not want to reinitialize in case the
            # data is not distributed IID as it could be the case that by chance no data owners were selected having
            # data on a specific centroid.
            if any(winner.return_deltas()) == 0.0 and num_reinitialized < args['num_reinitialize']:
                print("We should re-initialize the centroids because at least one centroid was not updated.")
                init_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims, n_clusters)
                centroids = init_centroids
                new_init_block = winner.build_block(centroids, re_init=True)
                winner.add_block(new_init_block)
                total_propagation_delay = winner.propagate_block(new_init_block)
                track_g_centroids = [init_centroids]
                total_broadcast_delay = 0
                for comm_member in committee_members_this_round:
                    if comm_member.online_switcher():
                        total_broadcast_delay += comm_member.request_to_download(new_init_block)
                for device in device_list:
                    # have each device reinitialize their kmeans model after reinitialization of centroids.
                    device.initialize_kmeans_model(n_dims=n_dims, n_clusters=device.elbow_method())

                # Delete all logged folders from previous communication rounds if a re-init occurred.
                re_init_event = True
                for i in range(1, comm_round + 2):
                    comm_round_file_path = f"{log_folder_path}/comm_{i}/"
                    if os.path.exists(log_folder_path):
                        shutil.rmtree(comm_round_file_path)
                num_reinitialized += 1
                comm_round = 0  # finally, reset communication round to 0.
            elif winner.request_final_block_verification():
                winner.add_block(winner.return_proposed_block())
                total_propagation_delay = winner.propagate_block(winner.return_proposed_block())
                # select a random committee member to request others to download the block after appendage.
                random.shuffle(committee_members_this_round)
                total_broadcast_delay = 0
                for comm_member in committee_members_this_round:
                    if comm_member.online_switcher():
                        total_broadcast_delay += comm_member.request_to_download(winner.return_proposed_block())
                if args['verbose']:
                    for block in device_list[-1].blockchain.get_chain_structure()[-2:]:
                        print(str(block), '\n')
                    print(f"Took {total_propagation_delay} seconds to propagate the winning block to each committee "
                          f"member, whereafter it took the committee members another {total_broadcast_delay} seconds "
                          f"to communicate the winning block with their peers.")
                track_g_centroids.append(winner.retrieve_global_centroids())
                is_malicious_node = "M" if winner.return_is_malicious() else "B"
                with open(f"{log_folder_path_comm_round}/round_{comm_round + 1}_info.txt", 'a') as file:
                    file.write(f"Updated global centroids are: {winner.retrieve_global_centroids()}.\n")
                    file.write(f"Deltas (Euclidean distance) between previous centroids and new centroids are: "
                               f"{winner.return_deltas()}.\n")
                    file.write(f"Winner's alignment was {is_malicious_node}. \n")
                if winner.return_stop_check():  # stop the global learning process.
                    print("Stopping condition met. Requesting peers to stop the global learning process...")
                    total_comm_rounds = args['num_comm']  # set comm_rounds at max rounds to stop the process.
                # Log updated centroids separately
                with open(f"{log_folder_path_comm_round}/updated_centroids.data", "wb") as file:
                    pickle.dump(winner.retrieve_global_centroids(), file)
        else:
            # check whether it holds for all leaders that any(leader.return_deltas()) == 0.0, then reinit.
            deltas_list = []
            for leader in leaders_this_round:
                deltas_list.append(leader.return_deltas())
            reinit = []
            for deltas in deltas_list:
                if any([delta == 0.0 for delta in deltas]):
                    reinit.append(True)
                else:
                    reinit.append(False)

            # if all leaders received no updates for a certain centroid, reinitialize.
            if all(reinit) and num_reinitialized < args['num_reinitialize']:
                print("We should re-initialize the centroids because at least one centroid was not updated.")
                init_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims, n_clusters)
                centroids = init_centroids
                # randomly pick a leader to perform the reinitialization.
                random.shuffle(leaders_this_round)
                chosen_leader = leaders_this_round[-1]
                new_init_block = chosen_leader.build_block(centroids, re_init=True)
                chosen_leader.add_block(new_init_block)
                total_propagation_delay = chosen_leader.propagate_block(new_init_block)
                track_g_centroids = [init_centroids]
                total_broadcast_delay = 0
                for comm_member in committee_members_this_round:
                    if comm_member.online_switcher():
                        total_broadcast_delay += comm_member.request_to_download(new_init_block)
                for device in device_list:
                    # have each device reinitialize their kmeans model after reinitialization of centroids.
                    device.initialize_kmeans_model(n_dims=n_dims, n_clusters=device.elbow_method())

                # Delete all logged folders from previous communication rounds if a re-init occurred.
                re_init_event = True
                for i in range(1, comm_round + 2):
                    comm_round_file_path = f"{log_folder_path}/comm_{i}/"
                    if os.path.exists(log_folder_path):
                        shutil.rmtree(comm_round_file_path)
                num_reinitialized += 1
                comm_round = 0  # finally, reset communication round to 0.

            num_rounds_no_winner += 1

        comm_round_time_taken = time.time() - comm_round_start_time  # total time of the comm round.
        print(f"Time taken this communication round: {comm_round_time_taken} seconds.")
        time_taken_per_round.append(comm_round_time_taken)
        parallel_time_estimate = role_assignment_time + max_local_update_time + max_aggr_time + max_proposal_time + \
                                 latest_vote_cast_time + block_completion_time + total_propagation_delay + \
                                 total_broadcast_delay
        estimate_wo_role_assignment = parallel_time_estimate - role_assignment_time
        print(f"Estimate time spent if all devices ran in parallel (real distributed system): "
              f"{parallel_time_estimate} seconds.")
        print(f"Limited to time spent by devices (i.e. excluding role assignment): {estimate_wo_role_assignment} "
              f"seconds.")
        print(max_local_update_time, max_aggr_time, max_proposal_time, latest_vote_cast_time, block_completion_time,
              total_propagation_delay, total_broadcast_delay)
        est_time_taken_parallel_per_round.append(parallel_time_estimate)

        if not re_init_event:
            with open(f"{log_folder_path_comm_round}/round_{comm_round + 1}_info.txt", 'a') as file:
                file.write(f"Time spent this communication round: {comm_round_time_taken} seconds.\n")
                file.write(f"Estimate time taken if devices ran in parallel: {parallel_time_estimate} seconds. \n")
                file.write(f"Estimate time without role assignment: {estimate_wo_role_assignment} seconds. \n")
                # file.write(f"Estimate time spent if all devices ran in parallel (real distributed system): "
                #            f"{parallel_time_estimate} seconds. \n")
                silhouette_scores = []
                # Need to treat the dataset as a global dataset to be able to compare with centralized and
                # federated k-means
                global_dataset = []
                longest_chain_len = 0
                device_longest_chain = device_list[-1]
                for device in device_list:
                    if device.return_blockchain_obj().get_chain_length() > longest_chain_len:
                        longest_chain_len = device.return_blockchain_obj().get_chain_length()
                        device_longest_chain = device
                    silhouette_scores.append(device.validate_update(device.retrieve_global_centroids()))
                    global_dataset.append(device.dataset)
                file.write(f"Current average silhouette score across all devices: "
                           f"{sum(silhouette_scores) / len(silhouette_scores)}. \n")
                # Log the silhouette score if we were treating the combined datasets as a single dataset.
                global_centroids = device_longest_chain.retrieve_global_centroids()
                global_dataset = np.asarray([record for sublist in global_dataset for record in sublist])
                global_model = cluster.KMeans(n_clusters=global_centroids.shape[0], init=global_centroids, n_init=1,
                                              max_iter=1)
                cluster_labels = global_model.fit_predict(global_dataset)
                file.write(f"Combining the local datasets into one produces a silhouette score of: "
                           f"{silhouette_score(global_dataset, cluster_labels)} \n")
                file.write(f"Combining the local datasets into one produces a Davies-Bouldin score of: "
                           f"{davies_bouldin_score(global_dataset, cluster_labels)} \n")
                file.write(f"Number of leaders this round: {len(leaders_this_round)}, \n"
                           f"Number of committee members this round: {len(committee_members_this_round)}, \n"
                           f"Number of data owners this round: {len(data_owners_this_round)}. \n")

            # log silhouette per device, also do this for aggregates at committee members.
            for device in device_list:
                silhouette_this_round = device.validate_update(device.retrieve_global_centroids())
                with open(f"{log_folder_path_comm_round}/silhouette_round_{comm_round + 1}.txt", 'a') as file:
                    # N.B. whether the node is malicious does not seem relevant to be logged here.
                    is_malicious_node = "M" if device.return_is_malicious() else "B"
                    file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}: "
                               f"{silhouette_this_round} \n")
                with open(f"{log_folder_path_comm_round}/silhouette_aggr_round_{comm_round + 1}.txt", 'a') as file:
                    if device.return_role() == 'committee' and len(device.updated_centroids) > 0:
                        file.write(f"{device.return_idx()} obtained aggregate {device.updated_centroids} achieving a "
                                   f"silhouette score of {device.validate_update(np.asarray(device.updated_centroids))} "
                                   f"\n")
                    elif device.return_role() == 'committee':
                        file.write(f"{device.return_idx()} obtained aggregate {device.updated_centroids}. \n")

            comm_round += 1  # finally, increment the round nr.

    # Log the blockchain after global learning is done, N.B. already did + 1
    with open(f"{bc_folder}/round_{comm_round}.txt", 'a') as file:
        blocks = [str(block) for block in device_list[-1].return_blockchain_obj().get_chain_structure()]
        for block in blocks:
            file.write(str(block) + "\n\n")

    # Log the blockchain as .json, so that it can be (easily) read and converted back to blockchain object again.
    blocks = device_list[-1].return_blockchain_obj().get_chain_structure()
    for block in blocks:
        with open(f"{bc_folder}/round_{comm_round}_block_{block.index}.json", 'a') as file:
            json_block = block.toJSON()
            file.write(json_block)

    print(f"Total time spent performing {total_comm_rounds} rounds: {sum(time_taken_per_round)} seconds.")
    print(f"Estimate if all devices ran in parallel during {total_comm_rounds} rounds: "
          f"{sum(est_time_taken_parallel_per_round)} seconds.")

    # Plot time taken per round.
    # fig1, ax1 = plt.subplots(1, 1)
    # ax1.plot(range(1, total_comm_rounds + 1), time_taken_per_round)
    # ax1.set_title('Time taken per communication round')
    # ax1.set_xlabel('Round number')
    # ax1.set_ylabel('Time taken (s)')
    # ax1.set_ylim([0.0, 4])
    # fig1.savefig(fname=f"{log_folder_path}/time_per_round.png", dpi=600, bbox_inches='tight')
    # plt.show()

    # Same thing, but for the estimate of time it would take if this were a real distributed system.
    # fig2, ax2 = plt.subplots(1, 1)
    # ax2.plot(range(1, total_comm_rounds + 1), est_time_taken_parallel_per_round)
    # ax2.set_title('Estimate of time taken (real distributed system)')
    # ax2.set_xlabel('Round number')
    # ax2.set_ylabel('Time taken (s)')
    # ax2.set_ylim([0.0, 4])
    # fig2.savefig(fname=f"{log_folder_path}/est_parallel_time_taken.png", dpi=600, bbox_inches='tight')
    # plt.show()

    # Plot data accompanied by the global centroids over time. N.B. How to show time progression for global centroids?
    print(f"Total number of centroids recorded: {len(track_g_centroids)}. \n"
          f"Should be the same as chain length (if not reinitialized): "
          f"{device_list[0].return_blockchain_obj().get_chain_length()}.")
    fig3, ax3 = plt.subplots(1, 1)
    for device in device_list:
        ax3.scatter(device.dataset[:, 0], device.dataset[:, 1], color='green', s=30, alpha=.3)
    colors = cm.nipy_spectral(cluster_labels.astype(float) / len(track_g_centroids))
    # Plot the initial centroids separately.
    for i in range(len(track_g_centroids[0])):
        ax3.scatter(track_g_centroids[0][i][0], track_g_centroids[0][i][1], marker='D', color=colors[i], s=60,
                    edgecolors='k')

    for i in range(1, len(track_g_centroids) - 1):
        for j in range(len(track_g_centroids[0])):
            ax3.scatter(track_g_centroids[i][j][0], track_g_centroids[i][j][1], color=colors[j])

    # Plot the last centroids separately.
    for i in range(len(track_g_centroids[-1])):
        ax3.scatter(track_g_centroids[-1][i][0], track_g_centroids[-1][i][1], marker='*', color=colors[i], s=100,
                    edgecolors='k')
    fig3.savefig(fname=f"{log_folder_path}/clustering_over_time.png", dpi=600, bbox_inches='tight')
    # plt.show()

    # Final clustering visualization
    global_dataset = []
    longest_chain_len = 0
    device_longest_chain = device_list[-1]
    for device in device_list:
        if device.return_blockchain_obj().get_chain_length() > longest_chain_len:
            longest_chain_len = device.return_blockchain_obj().get_chain_length()
            device_longest_chain = device
        silhouette_scores.append(device.validate_update(np.asarray(device.retrieve_global_centroids())))
        global_dataset.append(device.dataset)
    global_centroids = np.asarray(device_longest_chain.retrieve_global_centroids())
    global_dataset = np.asarray([record for sublist in global_dataset for record in sublist])
    global_model = cluster.KMeans(n_clusters=global_centroids.shape[0], init=global_centroids, n_init=1,
                                  max_iter=1)
    cluster_labels = global_model.fit_predict(global_dataset)

    fig4, ax4 = plt.subplots(1, 1)
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax4.scatter(global_dataset[:, 0], global_dataset[:, 1], marker='.', s=30, lw=0, alpha=.7, c=colors, edgecolor='k')
    ax4.scatter(global_centroids[:, 0], global_centroids[:, 1], marker='o', c='white', s=200, edgecolor='k')
    for i, c in enumerate(global_centroids):
        ax4.scatter(c[0], c[1], marker='$%d$' % i, s=50, edgecolor='k')
    ax4.set_title("Visualization of clustered data.")
    ax4.set_xlabel("Feature space of the 1st feature")
    ax4.set_ylabel("Feature space of the 2nd feature")
    fig4.savefig(fname=f"{log_folder_path}/final_clustering.png", dpi=600, bbox_inches='tight')
    # plt.show()

    # Silhouette visualization (per cluster)
    fig5, ax5 = plt.subplots(1, 1)
    ax5.set_xlim([-.4, 1])
    ax5.set_ylim([0, len(global_dataset) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(global_dataset, cluster_labels)
    silhouette_samples = silhouette_samples(global_dataset, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette = silhouette_samples[cluster_labels == i]
        ith_cluster_silhouette.sort()

        ith_cluster_size = ith_cluster_silhouette.shape[0]
        y_upper = y_lower + ith_cluster_size

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax5.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7)
        # ax5.text(-0.05, y_lower + 0.5 * ith_cluster_size, str(i))

        # compute y_lower for next cluster
        y_lower = y_upper + 10

    ax5.set_title("Silhouette plot for the various clusters.")
    ax5.set_xlabel("Silhouette score")
    ax5.set_ylabel("Records sorted by silhouette score per cluster")
    ax5.axvline(x=silhouette_avg, color='red', linestyle='--', label='_nolegend_')  # vertical line to show avg
    ax5.set_yticks([])  # no ticks on y-axis
    ax5.set_xticks([-.4, -.2, 0, .2, .4, .6, .8, 1])
    ax5.legend(range(n_clusters + 1))
    fig5.savefig(fname=f"{log_folder_path}/silhouette_analysis.png", dpi=600, bbox_inches='tight')
    # plt.show()

    print(f"Total number of communication rounds: {total_comm_rounds}.")
    print(f"Proportion of rounds without a winner: {num_rounds_no_winner}/{total_comm_rounds}")
    print(f"Number of times the centroids were reinitialized: {num_reinitialized}")
