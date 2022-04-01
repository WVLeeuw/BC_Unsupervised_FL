import numpy as np
import time
import datetime as dt
import random
import copy
import pickle
from sys import getsizeof

from hashlib import sha256
from Crypto.PublicKey import RSA
from sklearn import cluster
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from collections import defaultdict

from blockchain import Blockchain
from block import Block
import KMeans
from utils import data_utils


class Device:
    def __init__(self, idx, ds, local_epochs, network_stability, committee_update_wait_time, committee_block_wait_time,
                 committee_threshold, contribution_lag, fed_avg, leader_wait_time, global_update_lag, equal_link_speed,
                 base_data_transmission_speed, equal_computation_power, is_malicious, check_signature, stop_condition,
                 bc=None, model=None):
        # Identifier
        self.idx = idx

        # Datasets
        self.dataset = ds[0]
        self.labels = ds[1]
        self.nr_records = len(self.dataset)  # used for FedAvg.

        # P2P network variables
        self.online = True
        self.peer_list = set()
        self.network_stability = network_stability
        self.equal_link_speed = equal_link_speed
        self.base_data_transmission_speed = base_data_transmission_speed
        self.equal_computation_power = equal_computation_power
        self.is_malicious = is_malicious

        self.devices_dict = None
        self.aio = False

        # Blockchain variables
        self.check_signature = check_signature
        if bc is not None:
            self.blockchain = copy.copy(bc)
        else:
            self.blockchain = Blockchain()

        # Set link speed and computation power
        if equal_link_speed:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed

        if equal_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.uniform(0.1, 4)

        # General parameters, not role-specific.
        self.role = None
        self.model = model
        self.elbow = None
        self.recent_centroids = []
        self.stop_condition = stop_condition
        # Keys
        self.modulus = None
        self.private_key = None
        self.public_key = None

        # Role-specific parameters.
        # Data owners
        self.local_epochs = local_epochs
        self.performance_this_round = float('-inf')
        self.local_update_time = 0
        self.local_total_epoch = 0
        self.local_centroids = []

        # Committee members
        self.associated_data_owners_set = set()
        self.contributions_this_round = {}
        # self.associated_leaders = set()
        self.obtained_local_centroids = []  # obtained local centroids from data owners.
        self.centroids_idxs_records = []  # list of tuples describing local centroids and their sender idx.
        self.obtained_updates_unordered = {}
        self.obtained_updates_arrival_order_queue = {}
        self.updated_centroids = []
        self.committee_update_wait_time = committee_update_wait_time
        self.committee_block_wait_time = committee_block_wait_time
        self.committee_threshold = committee_threshold
        self.contribution_lag = contribution_lag
        self.fed_avg = fed_avg
        self.committee_validation_time = 0
        self.committee_aggregation_time = 0
        self.committee_block_validation_time = 0
        self.candidate_blocks = []
        self.candidate_blocks_unordered = {}  # necessary for simulation, not reflective of real distributed system.
        self.candidate_blocks_ordered = {}
        self.received_propagated_block = None

        # Leaders
        self.leader_wait_time = leader_wait_time
        self.global_update_lag = global_update_lag
        self.seen_committee_idxs = set()
        self.feedback_dicts = []
        self.obtained_aggregated_unordered = {}
        self.obtained_aggregates_arrival_order_queue = {}
        self.new_centroids = []
        self.deltas = []  # Euclidean distances between previous centroids and new centroids.
        self.proposed_block = None
        self.global_update_time = 0
        self.proposal_time = 0
        self.received_votes = []
        self.stop_check = False

        # For malicious devices
        self.variance_of_noise = None or []

    ''' setters '''

    def generate_rsa_key(self):
        kp = RSA.generate(bits=1024)
        self.modulus = kp.n
        self.private_key = kp.d
        self.public_key = kp.e

    def set_devices_dict_and_aio(self, devices_dict, aio):
        self.devices_dict = devices_dict
        self.aio = aio

    def assign_data_role(self):
        self.role = "data owner"
        self.model = cluster.KMeans(n_clusters=3, max_iter=self.local_epochs)

    def assign_committee_role(self):
        self.role = "committee"
        self.generate_rsa_key()  # for voting (through signing block hashes)

    def assign_leader_role(self):
        self.role = "leader"
        self.generate_rsa_key()  # for signing blocks

    # if anything goes wrong during the learning process of a certain device, this resets their k-means model.
    def initialize_kmeans_model(self, n_dims, n_clusters=3, verbose=False):
        bounds = data_utils.obtain_bounds(self.dataset)
        bounds_per_dim = []
        for i in range(len(bounds[0])):
            bounds_per_dim.append([bounds[0][i], bounds[1][i]])

        init_centroids = KMeans.randomly_init_centroid_range(values=bounds_per_dim, n_dims=n_dims, repeats=n_clusters)
        self.model = cluster.KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, max_iter=self.local_epochs)
        self.model.fit(self.dataset)
        self.local_centroids = self.model.cluster_centers_

    ''' getters '''

    def return_idx(self):
        return self.idx

    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def reset_role(self):
        self.role = None

    def is_online(self):
        return self.online

    # To check whether a malicious device is correctly excluded (i.e. poor contribution and/or reputation).
    def return_is_malicious(self):
        return self.is_malicious

    def return_blockchain_obj(self):
        return self.blockchain

    def return_pk(self):
        return self.public_key

    def return_link_speed(self):
        return self.link_speed

    def return_computation_power(self):
        return self.computation_power

    def return_nr_records(self):
        return self.nr_records

    def return_fed_avg(self):
        return self.fed_avg

    # Returns the performance (using some performance measure) of a data owner for the current round
    def return_performance(self):
        return self.performance_this_round

    ''' functions '''

    ''' not role-specific '''

    def sign(self, msg):
        msg = str(msg).encode('utf-8')
        h = int.from_bytes(sha256(msg).digest(), byteorder='big')
        signature = pow(h, self.private_key, self.modulus)
        return signature

    def verify_signature(self, block):
        if self.check_signature:
            modulus = block.get_miner_pk()["modulus"]
            pub_key = block.get_miner_pk()["pub_key"]
            signature = block.get_signature()
            msg = str(copy.copy(block)).encode('utf-8')
            h = int.from_bytes(sha256(msg).digest(), byteorder='big')
            h_signed = pow(signature, pub_key, modulus)
            if h == h_signed:
                return True
            else:
                return False  # placeholder, may want different behavior here.
        return True

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device):  # if it is a single device
            self.peer_list.add(new_peers)
        else:  # list otherwise
            self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):  # if it is a single device
            self.peer_list.discard(peers_to_remove)
        else:  # list otherwise
            self.peer_list.difference_update(peers_to_remove)

    def online_switcher(self):
        cur_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.online = True
            # if the device has been offline, we update their peer list and resync the chain
            if cur_status == False:
                print(f"{self.idx} has come back online.")
                # update peer list
                self.update_peer_list()
                # resync chain
                self.resync_chain()
        else:
            self.online = False
            print(f"{self.idx} has gone offline.")
        return self.online

    def update_peer_list(self):
        print(f"{self.idx} ({self.role}) is updating their peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        # use online peers to build peer list further
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove self from peer_list if it has been added
        self.remove_peers(self)

        if old_peer_list == self.peer_list:
            print("Peer list has not been changed.")
        else:
            print("Peer list has been changed.")

    def register_in_network(self, check_online=False):
        if self.aio:
            self.add_peers(set(self.devices_dict.values()))
        else:
            potential_registrars = set(self.devices_dict.values())
            potential_registrars.discard(self)  # ensure self is not in the set of registrars
            # pick a registrar randomly
            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrars = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrars.add(registrar)
                    if not online_registrars:  # no potential registrars are online
                        return False
                    registrar = random.sample(online_registrars, 1)[0]
            # add registrar to peer list
            self.add_peers(registrar)
            # and copy registrar's peer list
            self.add_peers(registrar.return_peers())
            # have the registrar add the newly connected device
            registrar.add_peers(self)
            return True

    # describes how to sync after reconnecting or failed validity check
    def resync_chain(self):
        longest_chain = None
        updated_from_peer = None
        cur_chain_len = self.return_blockchain_obj().get_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_obj()
                if peer_chain.get_chain_length() > cur_chain_len:
                    if peer_chain.is_chain_valid():
                        print(f"A longer chain from {peer.return_idx()} with length {peer_chain.get_chain_length()} "
                              f"was found and deemed valid.")
                        cur_chain_len = peer_chain.get_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(f"A longer chain from {peer.return_idx()} was found but could not be validated.")
        if longest_chain:
            # compare difference between chains
            longest_chain_struct = longest_chain.get_chain_structure()
            self.return_blockchain_obj().replace_chain(longest_chain_struct)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
        print(f"Chain was not resynced. Length of {self.return_idx()}'s chain is "
              f"{self.return_blockchain_obj().get_chain_length()} blocks.")

    def retrieve_global_centroids(self):
        latest_block = self.obtain_latest_block()
        return latest_block.get_data()['centroids']

    def obtain_latest_block(self):
        return self.blockchain.get_most_recent_block()

    # called by miners after they receive a majority vote.
    def add_block(self, block_to_add):
        self.blockchain.mine(block_to_add)

    # in case a block is propagated (i.e. already mined and verified) and must be appended to a device's chain.
    def append_block(self, block_to_append):
        self.return_blockchain_obj().append_block(block_to_append)

    ''' data owner '''

    def local_update(self):
        start_time = time.time()
        # Retrieve the global centroids recorded in the most recent block.
        g_centroids = self.retrieve_global_centroids()
        # Build a new (local) model using the centroids.
        if self.elbow < g_centroids.shape[0]:  # less local centroids than global centroids
            nearest_g_centroids = []
            for i in range(self.elbow):  # enforces that the choice of local centroids is kept dependent on elbow.
                nearest_g_centroids.append(self.find_nearest_global_centroid(self.local_centroids[i]))
            self.model = cluster.KMeans(n_clusters=self.elbow, init=np.asarray(nearest_g_centroids), n_init=1,
                                        max_iter=self.local_epochs)
        else:  # simply copy the global centroids
            self.model = cluster.KMeans(n_clusters=g_centroids.shape[0], init=g_centroids, n_init=1,
                                        max_iter=self.local_epochs)

        self.model.fit(self.dataset)
        self.local_total_epoch += self.local_epochs  # add the nr of local epochs to the total nr of epochs ran.
        self.local_centroids = self.model.cluster_centers_
        # performance = self.model.inertia_
        self.local_update_time = (time.time() - start_time) / self.computation_power

    def return_local_update(self):
        return self.retrieve_local_centroids(), self.return_nr_records(), self.return_idx()

    def return_local_update_time(self):
        return self.local_update_time

    def malicious_local_update(self):
        assert self.is_malicious, "Attempted to provide a malicious local update on a device that is not malicious."
        start_time = time.time()
        min_vals, max_vals = data_utils.obtain_bounds(self.dataset)
        bounds = []
        for i in range(len(min_vals)):  # N.B. len(min_vals) should be equal to n_dims every single time.
            bounds.append([min_vals[i], max_vals[i]])
        # Supply garbage updates by randomly producing 'local' centroids.
        self.local_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims=len(bounds), repeats=self.elbow)
        self.local_update_time = (time.time() - start_time) / self.computation_power

    def retrieve_local_centroids(self):
        return self.local_centroids

    # Used to retrieve good choice of k for local model.
    def elbow_method(self):
        ks = range(2, 10)  # N.B. if this starts at 2, we return i + 1. If it starts at 1, we can return i.
        inertias = []
        elbow_found = False

        # For each k, perform 20 iterations and determine the resulting inertia.
        for k in ks:
            model = cluster.KMeans(n_clusters=k, max_iter=20)
            model.fit(self.dataset)
            inertias.append(model.inertia_)

        # If for some k the inertia is more than half that of the inertia for previous k, we have found our elbow.
        for i in range(1, len(inertias) - 1):
            if inertias[i] > .5 * inertias[i - 1]:
                self.elbow = i + 1
                elbow_found = True
                return self.elbow

        if not elbow_found:
            self.elbow = 2
            return 2

    # Used to reset variables at the start of a communication round (round-specific variables) for data owners
    def reset_vars_data_owner(self):
        self.performance_this_round = float('-inf')
        self.local_update_time = 0

    ''' committee member '''

    def return_update_wait_time(self):
        return self.committee_update_wait_time

    def return_block_wait_time(self):
        return self.committee_block_wait_time

    # retrieve local_centroids, nr_records (for FedAvg) and device_idx (to update contribution).
    def obtain_local_update(self, local_centroids, nr_records, data_owner_idx):
        self.obtained_local_centroids.append(local_centroids)  # for simple averaging.
        self.centroids_idxs_records.append((local_centroids, nr_records, data_owner_idx))

    # obtain the average silhouette score for the given local centroids.
    def validate_update(self, local_centroids):
        validation_time = time.time()
        self.model = cluster.KMeans(n_clusters=local_centroids.shape[0], init=local_centroids, n_init=1, max_iter=1)
        cluster_labels = self.model.fit_predict(self.dataset)
        silhouette_avg = silhouette_score(self.dataset, cluster_labels)
        # self.update_contribution(silhouette_avg)
        validation_time = (time.time() - validation_time) / self.computation_power
        self.committee_validation_time = validation_time
        return silhouette_avg

    def return_validation_time(self):
        return self.committee_validation_time

    # find the nearest global centroid given a centroid position.
    def find_nearest_global_centroid(self, centroid):
        g_centroids = self.retrieve_global_centroids()
        min_dist = float('inf')
        nearest_g_centroid = None  # initialize as []?
        for g_centroid in g_centroids:
            if euclidean(g_centroid, centroid) < min_dist:
                min_dist = euclidean(g_centroid, centroid)
                nearest_g_centroid = g_centroid
        return nearest_g_centroid

    # produce a list of local centroids per global centroid, to be aggregated.
    def match_local_with_global_centroids(self):
        print(f"{self.return_idx()} retrieved local centroids: {self.obtained_local_centroids}")
        updates_per_centroid = []
        for global_centroid in self.retrieve_global_centroids():
            to_aggregate = []
            for centroids in self.obtained_local_centroids:
                for centroid in centroids:
                    if np.array_equal(global_centroid, self.find_nearest_global_centroid(centroid)):
                        to_aggregate.append(centroid)
            print("Found " + str(len(to_aggregate)) + " local centroids with which to update the global centroid.")
            updates_per_centroid.append(to_aggregate)
        return updates_per_centroid

    # aggregate all local centroids that were matched to their respective global centroid to obtain the update.
    def aggr_updates(self, updates_per_centroid):
        aggregation_time = time.time()
        aggr_centroids = []
        for i in range(len(updates_per_centroid)):
            if not isinstance(updates_per_centroid[i], np.ndarray):  # convert to np.ndarray to use numpy functions.
                updates_per_centroid[i] = np.asarray(updates_per_centroid[i])
            if len(updates_per_centroid[i]) > 0:  # we can use np.mean function to take the averages.
                avgs = updates_per_centroid[i].mean(axis=0)  # taking simple average of 'columns' for now.
                aggr_centroids.append(avgs.tolist())
            else:  # committee member received no updates for this centroid
                aggr_centroids.append(self.retrieve_global_centroids()[i])  # should then put the global centroid.
        self.updated_centroids = np.asarray(aggr_centroids)
        aggregation_time = (time.time() - aggregation_time) / self.computation_power
        self.committee_aggregation_time = aggregation_time
        return np.asarray(aggr_centroids)

    # aggregate all local centroids per global centroid weighted by the no. records per data owner.
    def aggr_fed_avg(self):
        aggregation_time = time.time()
        aggr_centroids = []
        updates_weights = []
        nr_records_list = [local_update_tuple[1] for local_update_tuple in self.centroids_idxs_records]
        total_nr_records = sum(nr_records_list)

        # Obtain scaling factors --> no. records per data owner compared to average no. records.
        for local_update_tuple in self.centroids_idxs_records:
            curr_centroids = local_update_tuple[0]
            curr_nr_records = local_update_tuple[1]
            curr_idx = local_update_tuple[-1]  # to be used for validation step and contribution update.
            scaling_fac = self._obtain_scaling_factor(curr_nr_records, total_nr_records, len(nr_records_list))
            self.update_contribution((curr_idx, curr_centroids))
            updates_weights.append((curr_centroids, scaling_fac))

        to_aggregate_per_centroid = []
        # Match local updates with global centroids and attach their weights.
        for g_centroid in self.retrieve_global_centroids():
            to_aggregate = []
            associated_weights = []
            for update, weight in updates_weights:
                for centroid in update:
                    if np.array_equal(self.find_nearest_global_centroid(centroid), g_centroid):
                        to_aggregate.append(centroid)
                        associated_weights.append(weight)
            to_aggregate_per_centroid.append([to_aggregate, associated_weights])
        print([len(aggregate_per_centroid[0]) for aggregate_per_centroid in to_aggregate_per_centroid])
        print("Found the following updates per centroid, given associated weights: " + str(to_aggregate_per_centroid))

        # Compute the aggregated centroids using the local centroids and their associated weights per global centroid.
        for i in range(len(to_aggregate_per_centroid)):
            updates = to_aggregate_per_centroid[i][0]
            weights = to_aggregate_per_centroid[i][1]
            if len(updates) > 0:
                weighted_avgs = np.average(updates, axis=0, weights=weights)
                aggr_centroids.append(weighted_avgs)
            else:  # if there are no updates for a specific centroid, we simply copy the previous global centroid.
                aggr_centroids.append(self.retrieve_global_centroids()[i])
                # N.B. if a centroid does not retrieve updates, does this weigh into our decision whether we should
                # re-initialize the centroids?

        # set the local updated centroids to be the aggregated centroids.
        self.updated_centroids = np.asarray(aggr_centroids)
        aggregation_time = (time.time() - aggregation_time) / self.computation_power
        self.committee_aggregation_time = aggregation_time
        return np.asarray(aggr_centroids)

    def return_aggregation_time(self):
        return self.committee_aggregation_time

    def malicious_aggr_updates(self):
        assert self.is_malicious, "Attempted to compute aggregates maliciously on a device that is not malicious."
        start_time = time.time()
        # local_updates_obtained = [local_update_tuple[0] for local_update_tuple in self.centroids_idxs_records]

        # ToDo: determine whether it suffices to have the same behavior as malicious_local_update here.
        min_vals, max_vals = data_utils.obtain_bounds(self.dataset)
        bounds = []
        for i in range(len(min_vals)):  # N.B. len(min_vals) should be equal to n_dims every single time.
            bounds.append([min_vals[i], max_vals[i]])
        # Supply garbage updates by randomly producing 'local' centroids.
        self.updated_centroids = KMeans.randomly_init_centroid_range(bounds, n_dims=len(bounds), repeats=self.elbow)

        aggregation_time = (time.time() - start_time) / self.computation_power
        self.committee_aggregation_time = aggregation_time
        return self.updated_centroids

    # Obtain a factor based on the no. records compared to the average no. records per data owner.
    def _obtain_scaling_factor(self, datasize, totalsize, nr_data_owners):
        scaling_factor = datasize / (totalsize / nr_data_owners)
        return scaling_factor

    def return_aggregate_and_feedback(self):
        # print(f"Feedback this round from {self.return_idx()}: {self.contributions_this_round}.")
        return self.updated_centroids, self.contributions_this_round, self.return_idx()

    # Computes the contribution (in terms of gain in silhouette score) of a local update for a specific device.
    def update_contribution(self, idx_update):
        global_model_performance = self.validate_update(self.retrieve_global_centroids())
        device_idx = idx_update[0]
        local_centroids = idx_update[1]
        score = self.validate_update(local_centroids)
        self.contributions_this_round[device_idx] = score - global_model_performance

    def return_online_data_owners(self):
        online_data_owners = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "data owner":
                    online_data_owners.add(peer)
        return online_data_owners

    def return_online_leaders(self):
        online_leaders = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "leader":
                    online_leaders.add(peer)
        return online_leaders

    def add_device_to_associated_set(self, device_idx):
        assert self.role == 'committee'
        self.associated_data_owners_set.add(device_idx)

    def return_online_associated_data_owners(self):
        return self.associated_data_owners_set

    # ToDo: figure out whether to verify more block characteristics or just signature.
    def verify_block(self, block):
        if self.verify_signature(block):
            return True
        else:
            return False

    def add_to_candidate_blocks(self, block, arrival_time):
        self.candidate_blocks.append(block)
        self.candidate_blocks_unordered[arrival_time] = block

    def order_candidate_blocks(self):
        candidate_blocks = self.candidate_blocks_unordered
        self.candidate_blocks_ordered.update(dict(sorted(candidate_blocks.items())))
        return self.candidate_blocks_ordered

    # approve block effectively produces a vote for said block.
    # Easiest way to do this is to have a malicious leader produce an illegal block,
    # then print it if approve_block fails.
    def approve_block(self, block):
        start_time = time.time()
        msg = None
        recent_block_data = self.obtain_latest_block().get_data()

        rep_check = True
        old_pos_rep, old_neg_rep = recent_block_data['pos_reputation'], recent_block_data['neg_reputation']
        # check if, for any device, we encounter illegal reputation values.
        for device_idx_o, rep_val_o in old_neg_rep.items():
            for device_idx_n, rep_val_n in block.get_data()['neg_reputation'].items():
                if device_idx_o == device_idx_n:
                    if not rep_val_o - 2 < rep_val_n < rep_val_o + 2:
                        rep_check = False
        if rep_check:  # also check positive reputation values if we have not found any illegal values yet.
            for device_idx_o, rep_val_o in old_pos_rep.items():
                for device_idx_n, rep_val_n in block.get_data()['pos_reputation'].items():
                    if device_idx_o == device_idx_n:
                        if not rep_val_o - 2 < rep_val_n < rep_val_o + 2:
                            rep_check = False

        contr_check = True
        old_contribution = recent_block_data['contribution']
        # check if, for any device, we encounter illegal contribution values.
        for device_idx_o, contr_val_o in old_contribution.items():
            for device_idx_n, contr_val_n in block.get_data()['contribution'].items():
                if device_idx_o == device_idx_n:
                    # ToDo: figure out whether this is as tight a bound as we can get on contribution.
                    # Tightness of this bounds depends on the choice of contribution lag, which committee knows of.
                    if abs(contr_val_n - contr_val_o) >= 2:
                        contr_check = False

        # if the new block's centroids perform as well as or better than the previous block's centroids, we approve.
        validate_res = self.validate_update(block.get_data()['centroids']) >= self.validate_update(
            self.retrieve_global_centroids()) + self.committee_threshold

        if validate_res and rep_check and contr_check:
            msg = [self.sign(block.get_hash()), self.return_idx(), self.return_rsa_pub_key(), dt.datetime.now()]

        self.committee_block_validation_time = time.time() - start_time
        return msg

    def return_block_validation_time(self):
        return self.committee_block_validation_time

    def malicious_approve_block(self, block):
        msg = None
        # approve the block if it performs worse than the previous global centroids, for actual validation.
        if self.validate_update(block.get_data()['centroids']) < \
                self.validate_update(self.retrieve_global_centroids()):
            msg = [self.sign(block.get_hash()), self.return_idx(), self.return_rsa_pub_key(), dt.datetime.now()]
        return msg

    # accept the propagated block --> append it to our blockchain.
    def accept_propagated_block(self, propagated_block, sender_idx):
        sender_is_leader = False
        for peer in self.peer_list:
            if peer.return_idx() == sender_idx and peer.return_role == 'leader':
                sender_is_leader = True

        # If it is not a genesis block or a re-init block.
        if propagated_block.get_vote_serial() and self.approve_block(propagated_block):
            enough_votes = False
            len_vote_serial = len(propagated_block.get_vote_serial())
            len_vote = len(pickle.dumps(self.approve_block(propagated_block)[0]))
            num_committee = len([device for device in self.peer_list if device.return_role() == 'committee'])
            if len_vote_serial > (num_committee // 2) * len_vote:
                enough_votes = True

            if propagated_block.get_hash() != self.obtain_latest_block().get_hash() and sender_is_leader \
                    and enough_votes:
                self.append_block(propagated_block)
            else:
                print(f"The block was already added to {self.return_idx()}'s chain.")
        else:  # only check whether the hash is correct and the sender is a leader.
            if propagated_block.get_hash() != self.obtain_latest_block().get_hash() and sender_is_leader:
                self.append_block(propagated_block)
            else:
                print(f"The block was already added to {self.return_idx()}'s chain.")

    # Send a request to peers to download the propagated (and accepted) block from leader.
    def request_to_download(self, block_to_download):
        broadcast_delay = 0
        block_to_download_size = getsizeof(block_to_download)
        for peer in self.peer_list:
            if peer.online_switcher() and self.is_online():
                # check whether the block is valid and has not already been added to the chain for current peer.
                if peer.verify_block(block_to_download) and \
                        peer.obtain_latest_block().get_hash() != block_to_download.get_hash():
                    lower_link_speed = self.link_speed if self.link_speed < peer.return_link_speed() \
                        else peer.return_link_speed()
                    broadcast_delay += block_to_download_size / lower_link_speed
                    peer.append_block(block_to_download)
            else:
                print(f"Either {peer.return_idx()} or {self.return_idx()} is currently offline or the block was "
                      f"already added to {peer.return_idx()}'s chain.")
        return broadcast_delay

    # Sends the vote to the leader device.
    def send_vote(self, vote, device_idx):
        leader_found = False
        for peer in self.peer_list:
            if peer.return_idx() == device_idx and peer.return_role() == 'leader' and peer.online_switcher():
                peer.received_votes.append(vote)
                leader_found = True
        if not leader_found:
            print(f"{device_idx} is not a leader node.")

    def return_candidate_blocks(self):
        return self.candidate_blocks

    # Used to reset variables at the start of a communication round (round-specific variables) for committee members.
    def reset_vars_committee_member(self):
        self.associated_data_owners_set = set()
        self.feedback_dicts = {}
        self.committee_validation_time = 0
        self.committee_aggregation_time = 0
        self.committee_block_validation_time = 0
        self.obtained_local_centroids = []
        self.obtained_updates_unordered = {}
        self.updated_centroids = []
        self.centroids_idxs_records = []
        self.received_propagated_block = None
        self.candidate_blocks = []
        self.candidate_blocks_unordered = {}

    ''' leader '''

    def return_aggr_wait_time(self):
        return self.leader_wait_time

    # Computes the new global centroids given the aggregated centroids from committee members.
    def compute_update(self):
        global_update_computation_time = time.time()
        obtained_centroids = np.asarray(self.new_centroids)
        g_centroids = self.retrieve_global_centroids()
        print(f"Obtained centroids have shape: {obtained_centroids.shape}")
        updated_g_centroids = []
        # Compute aggregates.
        if len(obtained_centroids) > 0:
            for i in range(len(obtained_centroids[0])):
                updated_g_centroids.append(obtained_centroids[:, i].mean(axis=0))
        else:  # ToDo: placeholder for now, likely not what we want to happen. Just pass?
            return np.zeros(g_centroids.shape)

        # compute new global centroids (simple update step).
        new_g_centroids = []
        assert len(updated_g_centroids) == len(g_centroids), "Number of global centroids is not equal to number of " \
                                                             "aggregated centroids. "
        for i in range(len(updated_g_centroids)):
            new_g_centroids.append(self.global_update_lag * g_centroids[i] +
                                   (1 - self.global_update_lag) * updated_g_centroids[i])

        deltas = []
        for i in range(len(g_centroids)):
            deltas.append(euclidean(g_centroids[i], new_g_centroids[i]))
        stop_per_centroid = [delta < self.stop_condition for delta in deltas]
        print(deltas, all(stop_per_centroid))
        self.deltas = deltas

        if all(stop_per_centroid):
            self._broadcast_stop_request()

        global_update_computation_time = (time.time() - global_update_computation_time) / self.computation_power
        self.global_update_time = global_update_computation_time
        return np.asarray(new_g_centroids)

    def malicious_compute_update(self):
        assert self.is_malicious, "Attempted to compute a malicious global update on a device that is not malicious."
        start_time = time.time()
        actual_update = self.compute_update()
        prev_global_centroids = self.retrieve_global_centroids()
        # Obtain the difference between the updated centroids and the global centroids.
        diffs_per_centroid = []
        for i in range(len(prev_global_centroids)):
            diffs = []
            for j in range(len(prev_global_centroids[0])):
                diffs.append(actual_update[i][j] - prev_global_centroids[i][j])
            diffs_per_centroid.append(diffs)
        malicious_update = [glob_centroid - diff for glob_centroid, diff in
                            zip(prev_global_centroids, diffs_per_centroid)]
        global_update_computation_time = (time.time() - start_time) / self.computation_power
        self.global_update_time = global_update_computation_time
        return np.asarray(malicious_update)

    def return_global_update_time(self):
        return self.global_update_time

    def obtain_aggr_and_feedback(self, aggregated_centroids, feedback, committee_idx):
        self.new_centroids.append(aggregated_centroids)
        self.feedback_dicts.append(feedback)
        self.seen_committee_idxs.add(committee_idx)

    # Build a candidate block from the newly produced global centroids, computed contribution scores and provide
    # feedback on committee members by updating their reputation.
    def build_block(self, new_g_centroids, re_init=False):
        block_proposal_time = time.time()
        previous_hash = self.obtain_latest_block().get_hash()
        data = dict()
        data['centroids'] = new_g_centroids
        data['contribution'] = self.update_contribution_final()
        if re_init:  # reset contribution values.
            device_idxs = self.obtain_latest_block().get_data()['contribution'].keys()
            data['contribution'] = dict.fromkeys(device_idxs, 0)
        data['pos_reputation'], data['neg_reputation'] = self.update_reputation()
        block = Block(index=self.blockchain.get_chain_length() + 1, data=data, previous_hash=previous_hash,
                      miner_pubkey=self.return_rsa_pub_key(), produced_by=self.return_idx())
        block.set_signature(self.sign(block))
        self.proposed_block = block
        block_proposal_time = (time.time() - block_proposal_time) / self.computation_power
        self.proposal_time = block_proposal_time
        return block

    def return_proposed_block(self):
        return self.proposed_block

    def malicious_build_block(self, new_g_centroids):
        assert self.is_malicious, "Attempted to build a malicious block on a device that is not malicious."
        block_proposal_time = time.time()
        previous_hash = self.obtain_latest_block().get_hash()
        recent_block_data = self.obtain_latest_block().get_data()
        data = dict()
        data['centroids'] = new_g_centroids
        data['contribution'] = recent_block_data['contribution']  # simply copy previous contribution
        data['pos_reputation'] = recent_block_data['pos_reputation']  # idem for reputation
        data['neg_reputation'] = recent_block_data['neg_reputation']
        block = Block(index=self.blockchain.get_chain_length() + 1, data=data, previous_hash=previous_hash,
                      miner_pubkey=self.return_rsa_pub_key(), produced_by=self.return_idx())
        block.set_signature(self.sign(block))
        self.proposed_block = block
        block_proposal_time = (time.time() - block_proposal_time) / self.computation_power
        self.proposal_time = block_proposal_time
        return block

    def return_proposal_time(self):
        return self.proposal_time

    # Return the proposed block, if there is any, along with the time it took to build it.
    def broadcast_block(self):
        return self.proposed_block, self.global_update_time

    # Broadcast a request to stop the global learning process. Called only when the stop condition is met.
    def _broadcast_stop_request(self):
        print('Stopping condition met! Requesting peers to stop global learning process...')
        self.stop_check = True

    def return_deltas(self):
        return self.deltas

    def return_stop_check(self):
        return self.stop_check

    # Propagate the proposed block (received majority vote) to all committee members.
    def propagate_block(self, block_to_propagate):
        total_transmission_delay = 0
        propagated_block_size = getsizeof(block_to_propagate)
        committee = self.return_online_committee_members()
        for member in committee:
            if member.online_switcher():
                lower_link_speed = self.link_speed if self.link_speed < member.return_link_speed() \
                    else member.return_link_speed()
                total_transmission_delay += propagated_block_size / lower_link_speed
                member.accept_propagated_block(block_to_propagate, self.return_idx())
        return total_transmission_delay

    def return_online_committee_members(self):
        online_committee_members = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "committee":
                    online_committee_members.add(peer)
        return online_committee_members

    def return_received_votes(self):
        return self.received_votes

    # For block finalization, to prove that the leader has counted votes they add them to the proposed block before
    # appending.
    def serialize_votes(self):
        assert self.proposed_block is not None, f"{self.return_idx()} has not proposed a block yet."
        obtained_signatures = []
        vote_timestamps = []
        for vote in self.received_votes:
            obtained_signatures.append(vote[0])
            vote_timestamps.append(vote[-1])
        serialized_votes = pickle.dumps(obtained_signatures)
        self.proposed_block.set_vote_serial(serialized_votes)
        self.proposed_block.set_block_time(max(vote_timestamps))
        self.proposed_block.set_signature(self.sign(self.proposed_block))

    # Request verification on the proposed block by committee members.
    # Returns True only if all committee members verified the block.
    def request_final_block_verification(self):
        assert self.proposed_block is not None, f"{self.return_idx()} has not proposed a block yet."
        comm_members = self.return_online_committee_members()
        verify_results = []
        for comm_member in comm_members:
            verify_results.append(comm_member.verify_block(self.proposed_block))
        return all(verify_results)  # returns False if any verification has failed.

    # checks for each committee member whether they successfully executed communication steps and if so,
    # assigns positive feedback. Negative otherwise.
    def update_reputation(self):
        committee_members_idxs = [peer.return_idx() for peer in self.peer_list if peer.return_role() == 'committee']
        updated_pos_reputation = copy.copy(self.obtain_latest_block().get_data()['pos_reputation'])
        updated_neg_reputation = copy.copy(self.obtain_latest_block().get_data()['neg_reputation'])

        for member_idx in self.seen_committee_idxs:
            if member_idx in updated_pos_reputation:  # check if key exists
                updated_pos_reputation[member_idx] = updated_pos_reputation.get(member_idx, 1) + 1
            else:  # first time we encounter the device
                # initialized at one, plus one successful interaction, for clarity
                updated_pos_reputation[member_idx] = 1 + 1

        for member_idx in committee_members_idxs:
            if member_idx not in self.seen_committee_idxs:
                if member_idx in updated_neg_reputation:
                    updated_neg_reputation[member_idx] = updated_neg_reputation.get(member_idx, 1) + 1
                else:  # first time we encounter the device
                    # initialized at one, plus one successful interaction, for clarity
                    updated_neg_reputation[member_idx] = 1 + 1

        return updated_pos_reputation, updated_neg_reputation

    # aggregates the contribution values found per device for each committee member and returns the aggregated
    # contribution scores.
    def update_contribution_final(self):
        # Loop through feedback_dicts and aggregate the contributions found for each key into a new dictionary,
        # to be added to the proposed block.
        contribution_update_time = time.time()
        final_contr = copy.copy(self.obtain_latest_block().get_data()['contribution'])
        beta = self.contribution_lag

        # produce a dictionary having the contribution values per device idx during this learning round.
        contr_by_idx = defaultdict(list)
        for contr_dict in self.feedback_dicts:
            for device_idx, contr in contr_dict.items():
                contr_by_idx[device_idx].append(contr)

        # obtain the average contribution computed during this learning round.
        final_contr_comp = {device_idx: sum(contributions) / len(contributions)
                            for device_idx, contributions in contr_by_idx.items()}

        # recall: C(local t) = beta * (L(local) - L(global)) + (1-beta) * C(local t-1)
        # where we consider L(.) to be the silhouette score. Positive C if it improves, negative C if it worsens.
        for device_idx in final_contr_comp.keys():
            if device_idx in final_contr:
                final_contr[device_idx] = (1-beta) * final_contr_comp[device_idx] + beta * final_contr[device_idx]
            else:
                final_contr[device_idx] = final_contr_comp[device_idx]  # first time we encounter the device.

        contribution_update_time = (time.time() - contribution_update_time) / self.computation_power
        return final_contr

    # Used to reset variables at the start of a communication round (round-specific variables) for leaders.
    def reset_vars_leader(self):
        self.seen_committee_idxs = set()
        self.feedback_dicts = []
        self.new_centroids = []
        self.proposed_block = None
        self.proposal_time = 0
        self.global_update_time = 0
        self.received_votes = []
        self.obtained_aggregated_unordered = {}
        self.obtained_aggregates_arrival_order_queue = {}
        self.stop_check = False


# Class to define and build each Device as specified by the parameters supplied. Returns a list of Devices.
class DevicesInNetwork(object):
    def __init__(self, is_iid, num_devices, num_malicious, local_epochs, network_stability, committee_update_wait_time,
                 committee_block_wait_time, committee_threshold, contribution_lag, fed_avg, leader_wait_time,
                 global_update_lag, equal_link_speed, data_transmission_speed, equal_computation_power, check_signature,
                 stop_condition, bc=None, dataset=None):
        self.dataset = dataset
        self.is_iid = is_iid
        self.num_devices = num_devices
        self.num_malicious = num_malicious
        self.network_stability = network_stability
        self.equal_link_speed = equal_link_speed
        self.equal_computation_power = equal_computation_power
        self.data_transmission_speed = data_transmission_speed
        # data owner
        self.local_epochs = local_epochs
        # committee
        self.committee_update_wait_time = committee_update_wait_time
        self.committee_block_wait_time = committee_block_wait_time
        self.committee_threshold = committee_threshold
        self.fed_avg = fed_avg
        self.contribution_lag = contribution_lag
        self.check_signature = check_signature
        self.stop_condition = stop_condition
        # leader
        self.leader_wait_time = leader_wait_time
        self.global_update_lag = global_update_lag
        # blockchain
        if bc is not None:
            self.blockchain = copy.copy(bc)
        else:
            self.blockchain = Blockchain()
        # divide data
        self.devices_set = {}
        self._dataset_allocation()

    # Split and allocate a given dataset among the devices in the network.
    def _dataset_allocation(self):
        # read dataset
        train_data, labels = data_utils.load_data(num_devices=self.num_devices, is_iid=self.is_iid,
                                                  dataset=self.dataset, samples=500)

        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(range(self.num_devices), self.num_malicious)

        # then create individual devices, each having their local dataset.
        for i in range(self.num_devices):
            is_malicious = False
            # divide the data equally
            local_dataset = [train_data[i], labels[i]]

            if i in malicious_nodes_set:
                is_malicious = True

            device_idx = f'device_{i + 1}'
            a_device = Device(device_idx, local_dataset, self.local_epochs, self.network_stability,
                              self.committee_update_wait_time, self.committee_block_wait_time, self.committee_threshold,
                              self.contribution_lag, self.fed_avg, self.leader_wait_time, self.global_update_lag,
                              self.equal_link_speed, self.data_transmission_speed, self.equal_computation_power,
                              is_malicious, self.check_signature, self.stop_condition, bc=self.blockchain)
            self.devices_set[device_idx] = a_device
            print(f"Creation of device having idx {device_idx} is done.")
        print("Creation of devices is done!")
