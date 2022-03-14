import hashlib
import math
import copy

import block
from block import Block
from blockchain import Blockchain
from device import Device, DevicesInNetwork
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
model_snapshots_folder = "models"

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
# parser.add_argument('-lc', '--num_local_centroids', type=int, default=3, help='number of centroids in locally trained'
#                                                                               ' models')
parser.add_argument('-le', '--num_local_epochs', type=int, default=1, help='number of epochs to perform for the '
                                                                           'acquisition of local update')
parser.add_argument('-fa', '--fed_avg', type=int, default=1, help='whether to use Federated Averaging')
parser.add_argument('-eps', '--epsilon', type=float, default=0.01,
                    help='threshold for the difference between the location of newly computed centroids and previous '
                         'global centroids s.t. when that difference is less than epsilon, the learning process ends.')
parser.add_argument('-gul', '--global_update_lag', type=float, default=.5,
                    help='parameter representing the lag of global model updates. Possible values between 0.0 and 1.0,'
                         'where a higher value represents more lag which in turn translates to new model updates '
                         'having less effect on the model update.')

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

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*',
                    help='number of devices assigned to the roles of data owner, committee member and leader '
                         'respectively')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='whether to check signatures, used to save '
                                                                          'time or to assume trust')
parser.add_argument('-aio', '--all_in_one_network', type=int, default=1,
                    help='whether to have all devices be aware of and connected to each other device in the network')
parser.add_argument('-cc', '--closely_connected', type=int, default=1,
                    help='whether to have each data owners be connected to all committee members or have the '
                         'connection be one to one.')

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
    if not os.path.isdir(model_snapshots_folder):
        os.mkdir(model_snapshots_folder)
    os.mkdir(f'{model_snapshots_folder}/{date_time}')

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
    open(f"{log_folder_path}/hello_world.txt", 'w').close()  # ToDo: change .txt name. Actually fil it during exe.

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
    while comm_round < args['num_comm']:
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
                if random.random() > .9 and len(chosen_catch_up) <= committee_members_to_assign // 4:
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
                if committee_members_to_assign > 0:
                    device.assign_committee_role()
                    committee_members_to_assign -= 1

            elif device.return_idx() in eligible_leaders_keys:
                if leaders_to_assign > 0:
                    device.assign_leader_role()
                    leaders_to_assign -= 1
            elif device.return_idx() in eligible_comm_keys:
                if committee_members_to_assign > 0:
                    device.assign_committee_role()
                    committee_members_to_assign -= 1

            # Ensure that all leader and committee member roles that were required are assigned.
            elif leaders_to_assign or committee_members_to_assign:
                if committee_members_to_assign > 0:
                    device.assign_committee_role()
                    committee_members_to_assign -= 1
                if leaders_to_assign > 0:
                    device.assign_leader_role()
                    leaders_to_assign -= 1

            # Done assigning committee members and leaders, adding them to their respective lists.
            if device.return_role() == "leader":
                leaders_this_round.append(device)
            if device.return_role() == "committee":
                committee_members_this_round.append(device)

            # ToDo: determine a value for the contribution less than 0.0 that is sensible to exclude devices for.
            # Check whether the remaining devices have high enough contribution to be selected as data owner.
            # N.B. They should not already have a role assigned.
            elif data_owners_to_assign > 0 and device.return_role() not in ['leader', 'committee'] \
                    and contr_vals[device.return_idx()] >= -.2:
                device.assign_data_role()
                data_owners_to_assign -= 1

            # Add data owners to the list of data owners.
            if device.return_role() == "data owner":
                data_owners_this_round.append(device)

            # finally, check whether the devices are online.
            device.online_switcher()

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
            device.local_update()
            if args['verbose']:
                local_centroids = device.retrieve_local_centroids()
                print(local_centroids)

            # Send the result to a committee member in the device's peer list.
            # Depending on the closeness of connections, put the data owner either in every committee member's
            # associated set, or only put them in a single committee member's associated set.
            if args['closely_connected']:
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
                eligible_comm_members[0].add_device_to_associated_set(device)

        # iii. committee members validate retrieved updates and aggregate viable results
        aggregated_local_centroids = []
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
                    else:
                        print(f"Data owner {data_owner.return_idx()} is unable to perform local update.")
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
                    else:
                        print(f"Data owner {data_owner.return_idx()} is unable to perform local update.")

            # validate local updates and aggregate usable local updates
            if comm_member.online_switcher():
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
                aggregated_local_centroids.append(aggr_centroids)  # not used anymore.

        # iv. committee members send updated centroids to every leader
        block_arrival_queue = {}
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
            proposed_g_centroids = leader.compute_update()
            block = leader.build_block(proposed_g_centroids)
            # after compute_update() and build_block(), both global_update_time and proposal_time are set.
            block.set_signature(leader.sign(block))  # after building the block, leaders sign it.
            if args['verbose']:
                print(str(block))

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
        votes = []
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
                                if vote:
                                    leader_idx = candidate_block.get_produced_by()
                                    print(f"Voted for {leader_idx}'s block")
                                    votes.append([vote, comm_member.return_idx(), leader_idx])
                            else:
                                print("Candidate block could not be verified. \n"
                                      "Block originates from device having idx: " +
                                      str(candidate_block.get_produced_by()))

        # Counting stage!
        # A vote is a pair [0] contains the message, [1] contains the committee idx, [2] contains the leader idx.
        winning_block = False
        winner = None
        for vote in votes:
            while not winning_block:
                msg = vote[0]
                comm_member_idx = vote[1]
                leader_idx = vote[2]
                for comm_member in committee_members_this_round:
                    if comm_member.return_idx() == comm_member_idx:
                        comm_member.send_vote(msg, leader_idx)
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
            if any(delta == 0 for delta in winner.return_deltas()):
                print("We should re-initialize the centroids because at least one centroid was not updated.")
                # Produce a new genesis block.
                data = dict()
                init_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims, n_clusters)
                centroids = init_centroids
                bc = Blockchain()
                bc.create_genesis_block(device_idxs=idxs, centroids=centroids)
                track_g_centroids = [init_centroids]
                for device in device_list:
                    # feed the created blockchain with genesis block to each device.
                    device.blockchain = copy.copy(bc)
                    device.initialize_kmeans_model(n_dims=n_dims, n_clusters=device.elbow_method())
                num_reinitialized += 1
                comm_round = 0  # finally, reset communication round to 0.
            elif winner.request_final_block_verification():
                winner.add_block(winner.proposed_block)
                # ToDo: consider link speeds here.
                winner.propagate_block(winner.proposed_block)
                # select a committee member to request others to download the block after appendage.
                # ToDo: consider link speeds here.
                committee_members_this_round[-1].request_to_download(winner.proposed_block)
                if args['verbose']:
                    for block in device_list[-1].blockchain.get_chain_structure()[-2:]:
                        print(str(block))
                track_g_centroids.append(winner.retrieve_global_centroids())
                if winner.return_stop_check():  # stop the global learning process.
                    print("Stopping condition met. Requesting peers to stop the global learning process...")
                    comm_round = args['num_comm']  # set comm_rounds at max rounds to stop the process.
                    # ToDo: ask whether this suffices for the simulation, or whether stop requests should be handled.
        else:
            for leader in leaders_this_round:
                # Code repetition for now. Can we write this into a function (i.e. def reinitialize())?
                # check whether there is a centroid which was not updated.
                if any(leader.return_deltas()) == 0.0:
                    print("We should re-initialize the centroids because at least one centroid was not updated.")
                    # Produce a new genesis block.
                    data = dict()
                    init_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims, n_clusters)
                    centroids = init_centroids
                    bc = Blockchain()
                    bc.create_genesis_block(device_idxs=idxs, centroids=centroids)
                    track_g_centroids = [init_centroids]
                    for device in device_list:
                        # feed the created blockchain with genesis block to each device.
                        device.blockchain = copy.copy(bc)
                        device.initialize_kmeans_model(n_dims=n_dims, n_clusters=device.elbow_method())
                    num_reinitialized += 1
                    comm_round = 0  # finally, reset communication round to 0.
            num_rounds_no_winner += 1

        comm_round_time_taken = time.time() - comm_round_start_time  # total time of the comm round.
        print(f"Time taken this communication round: {comm_round_time_taken} seconds.")
        time_taken_per_round.append(comm_round_time_taken)
        comm_round += 1

    # Plot time taken per round.
    plt.plot(range(1, total_comm_rounds + 1), time_taken_per_round)
    plt.xlabel('Round number')
    plt.ylabel('Time taken (s)')
    plt.ylim([0.25, 3])
    plt.show()

    # Plot data accompanied by the global centroids over time. N.B. How to show time progression for global centroids?
    for device in device_list:
        plt.scatter(device.dataset[:, 0], device.dataset[:, 1], color='green', alpha=.3)
    colors = ['purple', 'orange', 'cyan']
    for i in range(len(track_g_centroids) - 1):
        for j in range(len(track_g_centroids[0])):
            plt.scatter(track_g_centroids[i][j][0], track_g_centroids[i][j][1], color=colors[j])

    # Plot the last centroids separately.
    for i in range(len(track_g_centroids[-1])):
        plt.scatter(track_g_centroids[-1][i][0], track_g_centroids[-1][i][1], marker='*', color='k', s=100)
    plt.show()

    print(f"Total number of communication rounds: {total_comm_rounds}.")
    print(f"Proportion of rounds without a winner: {num_rounds_no_winner}/{total_comm_rounds}")
    print(f"Number of times the centroids were reinitialized: {num_reinitialized}")
