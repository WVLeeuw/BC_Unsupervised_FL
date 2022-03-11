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
parser.add_argument('-lwt', '--leader_wait_time', type=float, default=1.0,
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
    n_dims, n_clusters = 2, args['num_global_centroids']  # ToDo: fix hard coding the number of dimensions.

    min_vals, max_vals = data_utils.obtain_bounds_multiple(np.asarray(datasets))
    bounds = []
    for i in range(len(min_vals)):  # N.B. len(min_vals) should be equal to n_dims every single time.
        bounds.append([min_vals[i], max_vals[i]])
    print(bounds)

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
        device.initialize_kmeans_model(n_clusters=device.elbow_method())
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
                if random.random() > .9 and len(chosen_catch_up) <= committee_members_to_assign//4:
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

        # reset the device's role from previous round, then assign their new role.
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

            # Check whether the remaining devices have high enough contribution to be selected as data owner.
            # ToDo: determine a value for the contribution less than 0.0 that is sensible to exclude devices for.
            elif data_owners_to_assign and contr_vals[device.return_idx()] >= -.2:
                device.assign_data_role()
                data_owners_to_assign -= 1

            # Add all devices to a list of their respective roles.
            if device.return_role() == "data owner":
                data_owners_this_round.append(device)
            elif device.return_role() == "leader":
                leaders_this_round.append(device)
            elif device.return_role() == "committee":
                committee_members_this_round.append(device)

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
            device.local_update()  # ToDo: consider local computation power here.
            local_centroids = device.retrieve_local_centroids()
            if args['verbose']:
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
            # ToDo: consider link speeds here.
            for data_owner in comm_member.return_online_associated_data_owners():
                comm_member.obtain_local_update(data_owner.retrieve_local_centroids(), data_owner.return_nr_records(),
                                                data_owner.return_idx())

            # validate local updates and aggregate usable local updates
            # ToDo: consider local computation power (for validation) here.
            if comm_member.return_fed_avg():
                aggr_centroids = comm_member.aggr_fed_avg()
            else:
                updates_per_centroid = comm_member.match_local_with_global_centroids()
                aggr_centroids = comm_member.aggr_updates(updates_per_centroid)

            print(aggr_centroids)
            print(str(comm_member.validate_update(aggr_centroids)) +
                  " compared to previous global model performance of " +
                  str(comm_member.validate_update(comm_member.retrieve_global_centroids())))
            aggregated_local_centroids.append(aggr_centroids)

        # iv. committee members send updated centroids to every leader
        for leader in leaders_this_round:
            # ToDo: consider link speeds here.
            for comm_member in leader.return_online_committee_members():
                comm_member.send_centroids(leader)  # committee members send their aggregated updates.
                comm_member.send_feedback(leader)  # committee members send the feedback (in terms of contribution) \
                # given to data owners.

            # v. leaders build candidate blocks using the obtained centroids and send it to committee members for
            # approval.
            proposed_g_centroids = leader.compute_update()  # ToDo: consider local computation power here.
            block = leader.build_block(proposed_g_centroids)
            block.set_signature(leader.sign(block))  # after building the block, leaders sign it.
            if args['verbose']:
                print(str(block))

            # ToDo: consider link speeds here (and with it, the arrival time of candidate blocks at the committee).
            leader.broadcast_block(block)  # block is added to each online committee member's candidate blocks.

        # vi. committee members vote on candidate blocks by sending their vote to all committee members (signed)
        winning_block = False
        winner = None
        for comm_member in committee_members_this_round:
            assert len(comm_member.candidate_blocks) > 0, "Committee member did not retrieve any proposed blocks."
            print(str(comm_member.return_idx()) + " is now checking their candidate blocks.")
            print(str(len(comm_member.candidate_blocks)) + " blocks retrieved.")
            for candidate_block in comm_member.candidate_blocks:
                # check whether the signature on the block is valid, ignore the block otherwise.
                if comm_member.verify_signature(candidate_block):
                    print(f"Candidate block from {candidate_block.get_produced_by()} "
                          f"has been verified by {comm_member.return_idx()}")
                    centroids_to_check = candidate_block.get_data()['centroids']
                    # ToDo: consider local computation time here (for validation).
                    print("Using the newly proposed global centroids from the candidate block, an average "
                          "silhouette score of " + str(comm_member.validate_update(centroids_to_check)) +
                          " was found.")
                    # Determine whether the block should be voted for or not.
                    vote = comm_member.approve_block(candidate_block)
                    if vote:
                        leader_idx = candidate_block.get_produced_by()
                        print(f"Voted for {leader_idx}'s block")
                        comm_member.send_vote(vote, leader_idx)  # send vote to corresponding leader.

                        # after casting the vote, check whether the number of votes exceeds a majority.
                        for leader in leaders_this_round:
                            if leader.return_idx() == leader_idx:
                                if len(leader.return_received_votes()) > len(committee_members_this_round) // 2:
                                    winning_block = True
                                    winner = leader
                                    winner.serialize_votes()  # serialize votes and add result to the proposed block.
                                    print(f"{leader_idx} has received a majority vote for their block.")
                                    break

                else:
                    print("Candidate block could not be verified. \n"
                          "Block originates from device having idx: " + str(candidate_block.get_produced_by()))
            if winning_block:  # statement may be unnecessary or never reached, included anyway.
                break

        # vii. leader that obtained a majority vote append their block to their chain and propagate it to all
        # committee members. Afterward, committee members request their peers to download the winning block.

        # check whether there is a centroid which was not updated.
        if winning_block:
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
                    device.initialize_kmeans_model(n_clusters=device.elbow_method())
                num_reinitialized += 1
                comm_round = 0  # finally, reset communication round to 0.
            elif winner.request_final_block_verification():
                winner.add_block(winner.proposed_block)
                winner.propagate_block(winner.proposed_block)
                # select a committee member to request others to download the block after appendage.
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
                        device.initialize_kmeans_model(n_clusters=device.elbow_method())
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
    for i in range(len(track_g_centroids)):
        for j in range(len(track_g_centroids[0])):
            plt.scatter(track_g_centroids[i][j][0], track_g_centroids[i][j][1], color=colors[j])
    plt.show()

    print(f"Total number of communication rounds: {total_comm_rounds}.")
    print(f"Proportion of rounds without a winner: {num_rounds_no_winner}/{total_comm_rounds}")
    print(f"Number of times the centroids were reinitialized: {num_reinitialized}")
