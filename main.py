import hashlib
import math
import copy
import pickle
import shutil

import block
from block import Block
from blockchain import Blockchain
from device import Device, DevicesInNetwork
from sklearn import cluster
from sklearn.metrics import silhouette_score
import KMeans
from utils import data_utils
import matplotlib.pyplot as plt

import os
import sys
from sys import getsizeof
import argparse
import random
import time
from datetime import datetime
import numpy as np

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
                    help="threshold value for the difference in performance to determine whether to consider a local "
                         "update")
parser.add_argument('-lwt', '--leader_wait_time', type=float, default=0.0,
                    help="time window during which leaders wait for committee members to send their resulting "
                         "aggregate after they obtained the local updates from data owners. Wait time of 0.0 is "
                         "associated with no time limit.")
parser.add_argument('-rfc', '--resume_from_chain', type=str, default=None,
                    help='resume the global learning process from an existing blockchain from the path of a saved '
                         'blockchain. Only provide the date.')

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
            bc_save_path = f"{bc_folder}/{args['resume_from_chain']}"
            # ToDo: read file... split on '\n' as every whitespace signifies the 'border' between blocks.
            with open(bc_save_path) as f:
                chain = f.read()
            blocks = chain.split('\n')
            pass
        else:
            # feed the created blockchain with genesis block to each device.
            device.blockchain = copy.copy(bc)
            device.initialize_kmeans_model(n_dims=n_dims, n_clusters=device.elbow_method())
            # simulates peer registration, connects to some or all devices depending on 'all_in_one_network'.
            device.set_devices_dict_and_aio(devices_in_network.devices_set, args['all_in_one_network'])
            device.register_in_network()
            if args['verbose']:
                print(str(device.obtain_latest_block()))
    # remove the device if it is in its own peer list
    for device in device_list:
        device.remove_peers(device)

    # 7. build log files, to be filled during execution
    # ToDo: build more files with sensible names. Actually fill them during exe.
    # open(f"{log_folder_path}/hello_world.txt", 'w').close()

    # 8. run elbow method to make a sensible choice for global k.
    k_choices = []
    for device in device_list:
        k_choices.append(device.elbow_method())
    print("Average choice of k found: " + str(math.ceil(sum(k_choices) / len(k_choices))))
    print(math.ceil(sum(k_choices) / len(k_choices)) == args['num_global_centroids'])
    # check if avg equals supplied parameter for global centroids.

    # BCFL-KMeans starts here
    time_taken_per_round = []
    total_comm_rounds = 0
    comm_round = 0
    num_rounds_no_winner = 0
    num_reinitialized = 0
    while total_comm_rounds < args['num_comm']:
        # create log folder for communication round
        log_folder_path_comm_round = f"{log_folder_path}/comm_{comm_round}"
        if os.path.exists(log_folder_path_comm_round):
            print(f"Deleting {log_folder_path_comm_round} and creating a new one.")
            shutil.rmtree(log_folder_path_comm_round)
        os.mkdir(log_folder_path_comm_round)
        print(f"\nCommunication round {comm_round}.")

        comm_round_start_time = time.time()  # to keep track how long communication rounds take.
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
        latest_block_data = device_list[-1].obtain_latest_block().get_data()
        pos_rep = latest_block_data['pos_reputation']
        neg_rep = latest_block_data['neg_reputation']
        contr_vals = latest_block_data['contribution']
        eligible_comm_members = {}
        eligible_leaders = {}
        chosen_catch_up = []  # 10% probability that a device having (1, 1) reputation is chosen.
        for device in device_list:
            pos_count, neg_count = pos_rep[device.return_idx()], neg_rep[device.return_idx()]
            if pos_count == neg_count == 1:  # ToDo: check whether this is the only condition to be chosen to catch up.
                # assign committee role with 10% probability.
                if random.random() > .9 and len(chosen_catch_up) <= committee_members_to_assign // 3:
                    chosen_catch_up.append(device.return_idx())

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
            if device.return_idx() in chosen_catch_up:
                if committee_members_to_assign:
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

            # ToDo: re-evaluate contribution completely, as it goes to zero for every device as learning rounds increase
            # Check whether the remaining devices have high enough contribution to be selected as data owner.
            # N.B. They should not already have a role assigned.
            elif data_owners_to_assign > 0 and device.return_role() not in ['leader', 'committee'] \
                    and contr_vals[device.return_idx()] >= -.05:
                device.assign_data_role()
                data_owners_to_assign -= 1
            # Other devices that do not meet the contribution requirement can still attempt in 10% of cases.
            elif random.random() > 0.9 and not device.return_role():
                print(f"{device.return_idx()} has been chosen to provide an update even though their contribution "
                      f"and/or reputation is poor.")
                device.assign_data_role()

            # Add data owners to the list of data owners.
            if device.return_role() == "data owner":
                data_owners_this_round.append(device)

            # finally, check whether the devices are online.
            device.online_switcher()

        # log roles assigned for the devices, along with their reputation and contribution values.
        for device in device_list:
            # N.B. We can also keep track of how often a device is selected for a (specific) role while being malicious.
            role_assigned = device.return_role()
            contribution_val = device.obtain_latest_block().get_data()['contribution'][device.return_idx()]
            reputation_val = (device.obtain_latest_block().get_data()['pos_reputation'][device.return_idx()],
                              device.obtain_latest_block().get_data()['neg_reputation'][device.return_idx()])
            is_malicious_node = "M" if device.return_is_malicious() else "B"
            with open(f"{log_folder_path_comm_round}/role_assignment_comm_{comm_round}.txt", 'a') as file:
                file.write(f"{device.return_idx()} {is_malicious_node} assigned {device.return_role()} having "
                           f"contribution {contribution_val} and reputation {reputation_val}. \n")
            with open(f"{log_folder_path_comm_round}/malicious_devices_comm_{comm_round}.txt", 'a') as file:
                if device.return_is_malicious():
                    file.write(f"{device.return_idx()} assigned {device.return_role()}. \n")

        # ToDo: print more useful statistics here for debugging at start of round.
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
                    eligible_comm_members_filtered = eligible_comm_members[:len(committee_members_this_round)//2]
                    for comm_member in eligible_comm_members_filtered:
                        comm_member.add_device_to_associated_set(device)

        # iii. committee members validate retrieved updates and aggregate viable results
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
                        # check whether the time taken was less than the allowed time for local updating.
                        if local_update_total_time < comm_member.return_update_wait_time():
                            if comm_member.online_switcher():
                                comm_member.obtain_local_update(local_update, nr_records, data_owner_idx)
                                with open(f"{log_folder_path_comm_round}/silhouette_diffs_{comm_round}.txt",
                                          'a') as file:
                                    data_owner_silhouette = data_owner.validate_update(local_update)
                                    data_owner_is_malicious_node = "M" if data_owner.return_is_malicious() else "B"
                                    comm_member_silhouette = comm_member.validate_update(local_update)
                                    file.write(f"{comm_member_silhouette - data_owner_silhouette}, data owner "
                                               f"{data_owner.return_idx()} {data_owner_is_malicious_node} local "
                                               f"silhouette score was {data_owner_silhouette} while the committee "
                                               f"member {comm_member.return_idx()} obtained a silhouette score of "
                                               f"{comm_member_silhouette}. \n")
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
                        # finally, obtain the local update.
                        if comm_member.online_switcher():
                            comm_member.obtain_local_update(local_update, nr_records, data_owner_idx)
                            with open(f"{log_folder_path_comm_round}/silhouette_diffs_{comm_round}.txt", 'a') as file:
                                data_owner_silhouette = data_owner.validate_update(local_update)
                                data_owner_is_malicious_node = "M" if data_owner.return_is_malicious() else "B"
                                comm_member_silhouette = comm_member.validate_update(local_update)
                                file.write(f"{comm_member_silhouette-data_owner_silhouette}, data owner "
                                           f"{data_owner.return_idx()} {data_owner_is_malicious_node} local "
                                           f"silhouette score was {data_owner_silhouette} while the committee member "
                                           f"{comm_member.return_idx()} obtained a silhouette score of "
                                           f"{comm_member_silhouette}. \n")
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

                aggr_time = comm_member.return_aggregation_time()
                if args['verbose']:
                    print(aggr_centroids)
                    print(str(comm_member.validate_update(aggr_centroids)) +
                          " compared to previous global model performance of " +
                          str(comm_member.validate_update(comm_member.retrieve_global_centroids())))

        # iv. committee members send updated centroids to every leader
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
            with open(f"{log_folder_path_comm_round}/proposed_blocks_round_{comm_round}.txt", 'a') as file:
                is_malicious_node = "M" if leader.return_is_malicious() else "B"
                file.write(f"{leader.return_idx()} {is_malicious_node} "
                           f"proposed block: \n {leader.return_proposed_block()}. \n")

        # Determine the arrival order of proposed blocks.
        block_arrival_queue = {}
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
                                time_to_vote = comm_member.return_validation_time()
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

        # Counting stage!
        # A vote is a pair [0] contains the message, [1] contains the committee idx, [2] contains the leader idx.
        winning_block = False
        winner = None
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
                                print(f"{leader_idx} was the first to receive a majority vote for their block.")
                                break

        # vii. leader that obtained a majority vote append their block to their chain and propagate it to all
        # committee members. Afterward, committee members request their peers to download the winning block.
        if winning_block:
            # check whether there is a centroid which was not updated.
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
                num_reinitialized += 1
                comm_round = 0  # finally, reset communication round to 0.
                if args['verbose']:
                    for block in device_list[-1].blockchain.get_chain_structure()[-2:]:
                        print(str(block), '\n')
                    print(
                        f"Took {total_propagation_delay} seconds to propagate the winning block to each committee "
                        f"member, whereafter it took the committee members another {total_broadcast_delay} seconds "
                        f"to communicate the winning block with their peers.")
                break
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
                # ToDo: log stuff about the winning block
                with open(f"{log_folder_path_comm_round}/round_{comm_round}_info.txt", 'a') as file:
                    file.write(f"Updated global centroids are: {winner.retrieve_global_centroids()}.\n")
                    file.write(f"Deltas (Euclidean distance) between previous centroids and new centroids are:"
                               f"{winner.return_deltas()}.\n")
                if winner.return_stop_check():  # stop the global learning process.
                    print("Stopping condition met. Requesting peers to stop the global learning process...")
                    comm_round = args['num_comm']  # set comm_rounds at max rounds to stop the process.
                    # ToDo: ask whether this suffices for the simulation, or whether stop requests should be handled.
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
                num_reinitialized += 1
                comm_round = 0  # finally, reset communication round to 0.
                if args['verbose']:
                    for block in device_list[-1].blockchain.get_chain_structure()[-2:]:
                        print(str(block), '\n')
                    print(
                        f"Took {total_propagation_delay} seconds to propagate the winning block to each committee "
                        f"member, whereafter it took the committee members another {total_broadcast_delay} seconds "
                        f"to communicate the winning block with their peers.")
            num_rounds_no_winner += 1

        comm_round_time_taken = time.time() - comm_round_start_time  # total time of the comm round.
        print(f"Time taken this communication round: {comm_round_time_taken} seconds.")
        time_taken_per_round.append(comm_round_time_taken)

        # ToDo: figure out if there is any general stuff that should be logged here, every time.
        with open(f"{log_folder_path_comm_round}/round_{comm_round}_info.txt", 'a') as file:
            file.write(f"Time spent this communication round: {comm_round_time_taken} seconds.\n")
            silhouette_scores = []
            # Need to treat the dataset as a global dataset to be able to compare with centralized and federated k-means
            global_dataset = []
            global_centroids = device_list[-1].retrieve_global_centroids()
            for device in device_list:
                silhouette_scores.append(device.validate_update(device.retrieve_global_centroids()))
                global_dataset.append(device.dataset)
            file.write(f"Current average silhouette score across all devices: "
                       f"{sum(silhouette_scores)/len(silhouette_scores)}. \n")
            # Log the silhouette score if we were treating the combined datasets as a single dataset.
            global_dataset = np.asarray([record for sublist in global_dataset for record in sublist])
            global_model = cluster.KMeans(n_clusters=global_centroids.shape[0], init=global_centroids, n_init=1,
                                          max_iter=1)
            cluster_labels = global_model.fit_predict(global_dataset)
            file.write(f"Combining the local datasets into one produces a silhouette score of: "
                       f"{silhouette_score(global_dataset, cluster_labels)}. \n")

        # log silhouette per device, also do this for aggregates at committee members.
        for device in device_list:
            silhouette_this_round = device.validate_update(device.retrieve_global_centroids())
            with open(f"{log_folder_path_comm_round}/silhouette_round_{comm_round}.txt", 'a') as file:
                # N.B. whether the node is malicious does not seem relevant to be logged here.
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}: "
                           f"{silhouette_this_round}. \n")
            with open(f"{log_folder_path_comm_round}/silhouette_aggr_round_{comm_round}.txt", 'a') as file:
                if device.return_role() == 'committee' and len(device.updated_centroids) > 0:
                    file.write(f"{device.return_idx()} obtained aggregate {device.updated_centroids} achieving a "
                               f"silhouette score of {device.validate_update(np.asarray(device.updated_centroids))}. "
                               f"\n")
                elif device.return_role() == 'committee':
                    file.write(f"{device.return_idx()} obtained aggregate {device.updated_centroids}. \n")

        comm_round += 1  # finally, increment the round nr.

    # Log the blockchain after global learning is done.
    with open(f"{bc_folder}/round_{total_comm_rounds}.txt", 'a') as file:
        blocks = [str(block) for block in device_list[-1].return_blockchain_obj().get_chain_structure()]
        for block in blocks:
            file.write(str(block) + "\n\n")

    # Plot time taken per round.
    plt.plot(range(1, total_comm_rounds + 1), time_taken_per_round)
    plt.xlabel('Round number')
    plt.ylabel('Time taken (s)')
    plt.ylim([0.25, 3])
    plt.savefig(fname=f"{log_folder_path}/time_per_round.png")
    plt.show()

    # Plot data accompanied by the global centroids over time. N.B. How to show time progression for global centroids?
    print(f"Total number of centroids recorded: {len(track_g_centroids)}. \n"
          f"Should be the same as chain length (if not reinitialized): "
          f"{device_list[0].return_blockchain_obj().get_chain_length()}.")
    for device in device_list:
        plt.scatter(device.dataset[:, 0], device.dataset[:, 1], color='green', alpha=.3)
    colors = ['purple', 'orange', 'cyan']
    for i in range(len(track_g_centroids) - 1):
        for j in range(len(track_g_centroids[0])):
            plt.scatter(track_g_centroids[i][j][0], track_g_centroids[i][j][1], color=colors[j])

    # Plot the last centroids separately.
    for i in range(len(track_g_centroids[-1])):
        plt.scatter(track_g_centroids[-1][i][0], track_g_centroids[-1][i][1], marker='*', color='k', s=100)
    plt.savefig(fname=f"{log_folder_path}/clustering.png")
    plt.show()

    print(f"Total number of communication rounds: {total_comm_rounds}.")
    print(f"Proportion of rounds without a winner: {num_rounds_no_winner}/{total_comm_rounds}")
    print(f"Number of times the centroids were reinitialized: {num_reinitialized}")
