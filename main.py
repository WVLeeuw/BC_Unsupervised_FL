import hashlib
import math

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
# parser.add_argument('-B', '--batchsize', type=int, default=10, help='batch size used in local training')
parser.add_argument('-lr', '--learning_rate', type=float, default=.01, help='learning rate of k-means')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communication rounds')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='number of devices in the simulation')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help='number of malicious devices')
parser.add_argument('-iid', '--IID', type=int, default=1, help='whether to allocate data in iid setting')
parser.add_argument('-gc', '--num_global_centroids', type=int, default=3, help='number of centroids in the globally '
                                                                               'trained model')
parser.add_argument('-lc', '--num_local_centroids', type=int, default=3, help='number of centroids in locally trained '
                                                                              'models')
parser.add_argument('-le', '--num_local_epochs', type=int, default=1, help='number of epochs to perform for the '
                                                                           'acquisition of local update')

# BC attributes (consensus & committee parameters)
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0, help='threshold value of a difference in '
                                                                                  'accuracy with which to identify '
                                                                                  'malicious data owners')

# Additional BC attributes (to make entire process more efficient)
parser.add_argument('-cmt', '--committee_member_wait_time', type=float, default=0.0, help="default time window during "
                                                                                          "which committee members "
                                                                                          "wait for local updates to "
                                                                                          "be sent. Wait time of 0.0 "
                                                                                          "is associated with no time "
                                                                                          "limit.")
parser.add_argument('-cmh', '--committee_member_threshold', type=float, default=1.0, help="threshold value for the "
                                                                                          "difference in performance"
                                                                                          " to determine whether to "
                                                                                          "consider a local update")

# distributed system attributes
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds of a device being and '
                                                                                'staying online')
parser.add_argument('-els', '--equal_link_speed', type=int, default=1, help='used to simulate transmission delay. If '
                                                                            'set to 1, every device has equal link '
                                                                            'speed (bytes/sec). If set to 0, '
                                                                            'link speed is determined randomly.')
parser.add_argument('-dts', '--data_transmission_speed', type=float, default=70000.0, help="volume of data that is "
                                                                                           "transmitted per second "
                                                                                           "when -els == 1.")
parser.add_argument('-ecp', '--equal_computation_power', type=int, default=1, help='used to simulation computation '
                                                                                   'power. If set to 1, every device '
                                                                                   'has equal computation power. If '
                                                                                   'set to 0, computation power is '
                                                                                   'determined randomly.')

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help='number of devices assigned to the roles '
                                                                            'of data owner, committee member and '
                                                                            'committee leader respectively')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='whether to check signatures, used to save '
                                                                          'time or to assume trust')
parser.add_argument('-aio', '--all_in_one_network', type=int, default=1, help='whether to have all devices be aware '
                                                                              'of and connected to each other device '
                                                                              ' in the network')
parser.add_argument('-cc', '--closely_connected', type=int, default=1, help='whether to have all data owners be '
                                                                            'connected to all committee members or '
                                                                            'have the connection be one to one.')

if __name__ == '__main__':

    # get arguments
    args = parser.parse_args()
    args = args.__dict__

    # setup system variables
    latest_round_num = 0

    # set up from scratch
    # 0. create directory for log files
    if not os.path.isdir(log_folder_path):
        os.mkdir(log_folder_path)

    # 1. save arguments and parameters used
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

    # 3. get number of roles required in the network
    roles_requirement = args['hard_assign'].split(',')
    # determine roles to assign
    data_owners_needed = int(roles_requirement[0])
    committee_members_needed = int(roles_requirement[1])
    leaders_needed = int(roles_requirement[2])
    # then device.assign_data_role etc.

    # 4. check arguments eligibility
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

    # 5. Create devices in the network and their genesis block. Using dummy data for now.
    devices_in_network = DevicesInNetwork(is_iid=args['IID'], num_devices=num_devices, num_malicious=num_malicious,
                                          network_stability=args['network_stability'],
                                          committee_wait_time=args['committee_member_wait_time'],
                                          committee_threshold=args['committee_member_threshold'],
                                          equal_link_speed=args['equal_link_speed'],
                                          data_transmission_speed=args['data_transmission_speed'],
                                          equal_computation_power=args['equal_computation_power'],
                                          check_signature=args['check_signature'])
    device_list = list(devices_in_network.devices_set.values())

    # Extract the bounds on the data which was used for the creation of the devices.
    datasets = []
    for device in device_list:
        datasets.append(device.dataset)
    n_dims, n_clusters = 2, args['num_global_centroids']

    min_vals, max_vals = data_utils.obtain_bounds_multiple(np.asarray(datasets))
    bounds = []
    for i in range(len(min_vals)):  # N.B. len(min_vals) should be equal to n_dims every single time.
        bounds.append([min_vals[i], max_vals[i]])
    print(bounds)

    data = dict()
    init_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims, n_clusters)
    data['centroids'] = init_centroids
    bc = Blockchain()
    bc.create_genesis_block(data=data)

    # 6. register devices and initialize global parameters including genesis block.
    for device in device_list:
        # feed the created blockchain with genesis block to each device.
        device.blockchain = bc
        # initialize for each device their kmeans model. ToDo: utilize this to run the elbow method.
        device.initialize_kmeans_model(n_clusters=args['num_local_centroids'])
        # num_local_centroids should also be a parameter of the device constructor.
        # helper function simulating registration, effectively a setter in Device class
        device.set_devices_dict_and_aio(devices_in_network.devices_set, args['all_in_one_network'])
        # simulates peer registration, connects to some or all devices depending on 'all_in_one_network'.
        device.register_in_network()
        if args['verbose']:
            print(str(device.obtain_latest_block()))
    # remove the device if it is in its own peer list
    for device in device_list:
        device.remove_peers(device)

    # 7. build log files, to be filled during execution
    open(f"{log_folder_path}/hello_world.txt", 'w').close()
    # etc.

    # 8. run elbow method to make a sensible choice for global k.
    k_choices = []
    for device in device_list:
        # print("Sensible choice for # local clusters: " + str(device.elbow_method()))
        k_choices.append(device.elbow_method())
    print("Average choice of k found: " + str(math.ceil(sum(k_choices)/len(k_choices))))

    # BCFL-KMeans starts here
    for comm_round in range(latest_round_num + 1, args['num_comm'] + 1):
        # i. assign roles to devices dependent on contribution and reputation
        data_owners_to_assign = data_owners_needed
        committee_members_to_assign = committee_members_needed
        leaders_to_assign = leaders_needed

        data_owners_this_round = []
        committee_members_this_round = []
        leaders_this_round = []
        # for each device, draw a sample from its beta distribution (dependent on its reputation)
        beta_samples = []
        for device in device_list:
            # could put device.idx in the tuple rather than device.
            beta_samples.append(np.random.beta(device.reputation[0], device.reputation[1]))

        # ToDo: assign roles depending on the sampled reputation values and existing contribution values
        random.shuffle(device_list)  # for now, we just randomly shuffle them
        for device in device_list:
            if data_owners_to_assign:
                device.assign_data_role()
                data_owners_to_assign -= 1
            elif committee_members_to_assign:
                device.assign_committee_role()
                committee_members_to_assign -= 1
            elif leaders_to_assign:
                device.assign_leader_role()
                leaders_to_assign -= 1

            if device.return_role() == "data owner":
                data_owners_this_round.append(device)
            elif device.return_role() == "leader":
                leaders_this_round.append(device)
            elif device.return_role() == "committee":
                committee_members_this_round.append(device)

        # Reset variables for new comm round.
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

            # send the result to a committee member in the device's peer list.
            # currently, we simply put each data owner into each committee member's associated set.
            # if we put a handful of data owners into each committee member's associated set, this code is required.
            # alternatively, we perform a check_online for each device that is in the peer list, then check for role...
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
                # associate the device with only one committee member.
                eligible_comm_members[0].add_device_to_associated_set(device)

        # iii. committee members validate retrieved updates and aggregate viable results
        aggregated_local_centroids = []
        for comm_member in committee_members_this_round:
            global_centroids = comm_member.retrieve_global_centroids()
            # print(comm_member.return_idx() + " having associated data owners ...")
            updates_per_idx = []
            # alternatively, we can use return_online_data_owners per committee member.
            # ToDo: consider link speeds here.
            for data_owner in comm_member.return_online_associated_data_owners():
                # print(data_owner.return_idx())
                comm_member.local_centroids.append(data_owner.retrieve_local_centroids())
                updates_per_idx.append((data_owner.return_idx(), data_owner.retrieve_local_centroids()))  # tuple

            print(comm_member.return_idx() + " retrieved local centroids: " + str(comm_member.local_centroids))

            for idx_update in updates_per_idx:
                comm_member.update_contribution(idx_update)

            # validate local updates and aggregate usable local updates
            # ToDo: consider local computation power (for validation) here.
            usable_centroids = []  # not sure whether to use this or to simply check some performance measure
            updates_per_centroid = comm_member.match_local_with_global_centroids()
            aggr_centroids = comm_member.aggr_updates(updates_per_centroid)
            print(aggr_centroids)
            print(str(comm_member.validate_update(aggr_centroids)) +
                  " compared to previous global model performance of " +
                  str(comm_member.validate_update(comm_member.retrieve_global_centroids())))
            print(comm_member.compute_new_global_centroids(aggr_centroids))  # this statement can remain
            # aggregated_local_centroids.append(comm_member.compute_new_global_centroids(aggr_centroids))
            aggregated_local_centroids.append(aggr_centroids)

        # iv. committee members send updated centroids to every leader
        newly_proposed_centroids = []
        for leader in leaders_this_round:
            # check whether the committee members are (successfully) associating with the leader.
            # ToDo: consider link speeds here.
            for comm_member in leader.return_online_committee_members():
                comm_member.send_centroids(leader)
                # comm_members share their aggregates with the leader.
                # if no aggregate is obtained for some committee members, leaders give negative feedback.
                # otherwise, feedback is positive (only depends on retrieval, not quality of centroids).

            # v. leaders build candidate blocks using the obtained centroids and send it to committee members for
            # approval.
            proposed_g_centroids = leader.compute_update()  # ToDo: consider local computation power here.
            newly_proposed_centroids.append(proposed_g_centroids)
            block = leader.propose_block(proposed_g_centroids)
            block.set_signature(leader.sign(block))  # could do this within sign as msg is assumed to be a block.
            print(str(block))

            # ToDo: consider link speeds here (and with it, the arrival time of candidate blocks at the committee).
            leader.broadcast_block(block)  # block is added to each online committee member's candidate blocks.
            # idea: committee members loop through the blocks in the candidate blocks and check their validity.
            # afterwards, they check local performance with the newly proposed global centroids and vote accordingly.

        print(init_centroids, newly_proposed_centroids[0])
        # N.B. all leaders end up with the same proposed centroids as each committee member communicates with each
        # leader.

        # vi. committee members vote on candidate blocks by sending their vote to all committee members (signed)
        winning_block = False
        winner = None
        for comm_member in committee_members_this_round:
            assert len(comm_member.candidate_blocks) > 0, "Committee member did not retrieve any proposed blocks."
            print(str(comm_member.return_idx()) + " is now checking their candidate blocks.")
            print(str(len(comm_member.candidate_blocks)) + " blocks retrieved.")
            while not winning_block:
                for candidate_block in comm_member.candidate_blocks:
                    print(str(candidate_block.get_produced_by()))
                    # if the block signature is invalid, we ignore the block from the candidate blocks.
                    if comm_member.verify_block(candidate_block):
                        print("Candidate block has been verified by " + comm_member.return_idx())
                        centroids_to_check = candidate_block.get_data()['centroids']
                        # ToDo: consider local computation time here (for validation).
                        print("Using the newly proposed global centroids from the candidate block, an average "
                              "silhouette score of " + str(comm_member.validate_update(centroids_to_check)) +
                              " was found.")
                        # ToDo: check performance (validate proposed g_centroids) and vote accordingly.
                        # Voting may be done through comm_member.approve_block(candidate_block).
                        # A vote can either be cast if a certain threshold w.r.t performance is met, or,
                        # alternatively, a vote can be cast only for the block having the best performance.
                        vote = comm_member.approve_block(candidate_block)
                        # send vote to corresponding leader.
                        if vote:
                            leader_idx = candidate_block.get_produced_by()
                            print("Voted for " + leader_idx + "'s block")
                            comm_member.send_vote(vote, leader_idx)

                            # given device_idx, check whether the number of votes exceeds a majority.
                            for leader in leaders_this_round:
                                if leader.return_idx() == leader_idx:
                                    if len(leader.return_received_votes()) > len(committee_members_this_round) // 2:
                                        winning_block = True
                                        winner = leader
                                        print(f"{leader_idx} has received a majority vote for their block.")
                                        # break here.
                                        break
                    else:
                        print("Candidate block could not be verified. \n"
                              "Block originates from device having idx: " + str(candidate_block.get_produced_by()))

        # vii. leader that obtained a majority vote append their block to the chain and broadcast it to all devices
        # ToDo: implement winning_block boolean, meaning whether a committee member has found a winning block.
        # also consider whether multiple blocks can receive (positive) votes by a single committee member, or,
        # alternatively, have committee members vote on a single block each.
        if winning_block:
            # sign the hash of the best performing candidate block to produce a vote.
            winner.add_block(winner.proposed_block)
            winner.propagate_block(winner.proposed_block)
            # select a committee member to request others to download the block after propagation
            committee_members_this_round[-1].request_to_download(winner.proposed_block)
            for block in device_list[-1].blockchain.get_chain_structure():
                print(str(block))
            if winner.blockchain.get_chain_length() == 2:  # 100 comm rounds + genesis block.
                for device in device_list:
                    for i in range(len(device.dataset)-2):
                        plt.scatter(device.dataset[i][0], device.dataset[i][1], color='green', alpha=.3)
                plt.scatter(device_list[-1].dataset[:, 0], device_list[-1].dataset[:, 1], color='purple', alpha=.5)
                plt.scatter(device_list[-2].dataset[:, 0], device_list[-2].dataset[:, 1], color='cyan', alpha=.5)
                plt.scatter(init_centroids[:, 0], init_centroids[:, 1], color='orange')
                plt.scatter(winner.retrieve_global_centroids()[:, 0], winner.retrieve_global_centroids()[:, 1],
                            color='k')
                plt.show()
