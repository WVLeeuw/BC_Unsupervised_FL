import numpy as np
import time
import datetime as dt
import random
import copy
import pickle

from hashlib import sha256
from Crypto.PublicKey import RSA
from sklearn import cluster
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean

from blockchain import Blockchain
from block import Block
import KMeans
from utils import data_utils


class Device:
    def __init__(self, idx, ds, network_stability, committee_wait_time, committee_threshold, equal_link_speed,
                 base_data_transmission_speed, equal_computation_power, check_signature, stop_condition, bc=None,
                 model=None):
        # Identifier
        self.idx = idx

        # Trust values --> recorded on blockchain. Initialized in genesis block.

        # Datasets
        self.dataset = ds[0]
        self.labels = ds[1]
        self.nr_records = len(self.dataset)

        # P2P network variables
        self.online = True
        self.peer_list = set()
        self.network_stability = network_stability
        self.equal_link_speed = equal_link_speed
        self.base_data_transmission_speed = base_data_transmission_speed
        self.equal_computation_power = equal_computation_power
        self.black_list = set()

        self.devices_dict = None
        self.aio = False

        # BC variables
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
            self.computation_power = random.randint(0, 4)

        # Role-specific
        self.role = None
        self.model = model
        self.elbow = None
        self.recent_centroids = []
        self.stop_condition = stop_condition

        # Data owners
        self.performance_this_round = float('-inf')
        self.local_update_time = None
        self.local_total_epoch = 0
        self.local_centroids = []

        # Committee members
        self.associated_data_owners_set = set()
        self.performances_this_round = {}
        # self.associated_leaders = set()
        self.obtained_local_centroids = []  # obtained local centroids from data owners.
        self.centroids_idxs_records = []  # list of tuples describing local centroids and their sender idx.
        self.obtained_updates_unordered = {}
        self.obtained_updates_arrival_order_queue = {}
        self.updated_centroids = []
        self.committee_wait_time = committee_wait_time
        self.committee_threshold = committee_threshold
        self.committee_local_performance = float('-inf')  # this can be set after validation of new global centroids.
        self.candidate_blocks = []
        self.candidate_blocks_unordered = {}  # necessary for simulation, not reflective of real distributed system.
        self.candidate_blocks_arrival_order_queue = {}
        self.received_propagated_block = None

        # Leaders
        self.seen_committee_idxs = set()
        self.obtained_aggregated_unordered = {}
        self.obtained_aggregates_arrival_order_queue = {}
        self.new_centroids = []
        self.proposed_block = None
        self.received_votes = []
        # ToDo: figure out whether we also require to determine an arrival order of votes. Time-outs?
        # self.obtained_votes_unordered = {}
        # self.obtained_votes_arrival_order_queue = {}

        # Keys
        self.modulus = None
        self.private_key = None
        self.public_key = None

    ''' setters '''

    def generate_rsa_key(self):
        kp = RSA.generate(bits=1024)
        self.modulus = kp.n
        self.private_key = kp.d
        self.public_key = kp.e

    def set_devices_dict_and_aio(self, devices_dict, aio):
        self.devices_dict = devices_dict
        self.aio = aio

    # Function assigning the roles to devices with some probability and constraints. May be done elsewhere.
    def assign_role(self):
        # non-equal prob, depends on proportion worker:committee:leaders.
        pass

    # following assignments are for hard assignments
    def assign_data_role(self):
        self.role = "data owner"
        self.model = cluster.KMeans(n_clusters=3, max_iter=10)

    def assign_committee_role(self):
        self.role = "committee"
        self.generate_rsa_key()  # for voting (through signing block hashes)

    def assign_leader_role(self):
        self.role = "leader"
        self.generate_rsa_key()  # for signing blocks

    # if anything goes wrong during the learning process of a certain device, can reset
    def initialize_kmeans_model(self, n_clusters=3, verbose=False):
        init_centroids = KMeans.randomly_init_centroid_range(values=data_utils.obtain_bounds(self.dataset),
                                                             n_dims=len(self.dataset[0]), repeats=n_clusters)
        self.model = cluster.KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, max_iter=5)
        self.model.fit(self.dataset)
        self.local_centroids = self.model.cluster_centers_

    ''' getters '''

    def return_idx(self):
        return self.idx

    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_black_list(self):
        return self.black_list

    def return_role(self):
        return self.role

    def is_online(self):
        return self.online

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

    # Returns the performance (using some performance measure) of a data owner for the current round
    def return_performance(self):
        return self.performance_this_round

    # Returns the performances recorded from local updates by a committee member
    def return_performances(self):
        return self.performances_this_round

    ''' functions '''

    ''' not role-specific '''

    def sign(self, msg):  # msg is assumed to be a block.
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
                return False  # placeholder
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

    def switch_online_status(self):
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
                if self.resync_chain():
                    self.update_model_after_resync()
        else:
            self.online = False
            print(f"{self.idx} has gone offline.")
        return self.online

    # ToDo: print new list (if needed).
    def update_peer_list(self):
        print(f"{self.idx} - {self.role} is updating their peer list...")
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
        # code pertaining to blacklisted or otherwise untrusted devices goes here.
        potentially_malicious_peers = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potentially_malicious_peers.add(peer)
        # remove potentially malicious users
        self.remove_peers(potentially_malicious_peers)

        # N.B. could print the resulting peer list here (for debugging)
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
            # pick a registrar
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
            # and copy registrars peer list
            self.add_peers(registrar.return_peers())
            # have the registrar add the newly connected device
            registrar.add_peers(self)
            return True

    # describes how to sync after failed validity or fork
    def resync_chain(self):
        longest_chain = None
        updated_from_peer = None
        cur_chain_len = self.return_blockchain_obj().get_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_obj()
                if peer_chain.get_chain_length() > cur_chain_len:
                    if peer_chain.is_chain_valid():
                        print(f"A longer chain from {peer.return_idx()} with length {peer_chain.return_chain_length()} "
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
            return True
        print("Chain could not be resynced.")
        return False

    def update_model_after_resync(self):
        g_centroids = self.retrieve_global_centroids()
        self.model = cluster.KMeans(n_clusters=g_centroids.shape[0], init=g_centroids,
                                    n_init=1, max_iter=10)

    def retrieve_global_centroids(self):
        latest_block = self.obtain_latest_block()
        return latest_block.get_data()['centroids']

    # called by miners after they receive a majority vote.
    def add_block(self, block_to_add):
        self.blockchain.mine(block_to_add)

    # in case a block is propagated (i.e. already mined and verified) and must be appended to a device's chain.
    def append_block(self, block_to_append):
        self.return_blockchain_obj().append_block(block_to_append)

    def obtain_latest_block(self):
        return self.blockchain.get_most_recent_block()

    ''' data owner '''

    def local_update(self):
        start_time = time.time()
        # Retrieve the global centroids recorded in the most recent block.
        g_centroids = self.retrieve_global_centroids()
        # Build a new (local) model using the centroids.
        if self.elbow < g_centroids.shape[0]:  # less local centroids than global centroids
            nearest_g_centroids = []
            for centroid in self.local_centroids:
                nearest_g_centroids.append(self.find_nearest_global_centroid(centroid))
            self.model = cluster.KMeans(n_clusters=self.elbow, init=np.asarray(nearest_g_centroids), n_init=1,
                                        max_iter=5)
        else:  # simply copy the global centroids
            self.model = cluster.KMeans(n_clusters=g_centroids.shape[0], init=g_centroids, n_init=1, max_iter=5)

        self.model.fit(self.dataset)
        self.local_total_epoch += 5  # 5 local iterations were done this round.
        self.local_centroids = self.model.cluster_centers_
        # performance = self.model.inertia_
        self.local_update_time = time.time() - start_time

    def retrieve_local_centroids(self):
        return self.local_centroids

    # Used to retrieve good choice of k for local model.
    def elbow_method(self):
        ks = range(2, 10)  # N.B. if this starts at 2, we return i + 1. If it starts at 1, we can return i.
        inertias = []

        for k in ks:
            model = cluster.KMeans(n_clusters=k, max_iter=20)
            model.fit(self.dataset)
            inertias.append(model.inertia_)

        for i in range(1, len(inertias)-1):
            if inertias[i] > .5 * inertias[i-1]:
                self.elbow = i + 1
                return i + 1

    # Used to reset variables at the start of a communication round (round-specific variables) for data owners
    def reset_vars_data_owner(self):
        self.performance_this_round = float('-inf')
        self.local_update_time = None

    ''' committee member '''

    # ToDo: use a list of tuples to determine new contribution value for data_owner_idx.
    def obtain_local_update(self, local_centroids, nr_records, data_owner_idx):
        # retrieve local_centroids, nr_records (FedAvg) and device_idx (contribution).
        self.obtained_local_centroids.append(local_centroids)
        self.centroids_idxs_records.append((local_centroids, nr_records, data_owner_idx))

    # ToDo: rewrite s.t. we also take the device idx for update_contribution.
    # in other words, we want validate_update to call update_contribution, which requires device idx.
    def validate_update(self, local_centroids):
        self.model = cluster.KMeans(n_clusters=local_centroids.shape[0], init=local_centroids, n_init=1, max_iter=1)
        cluster_labels = self.model.fit_predict(self.dataset)
        silhouette_avg = silhouette_score(self.dataset, cluster_labels)
        # self.update_contribution(silhouette_avg)
        return silhouette_avg

    def find_nearest_global_centroid(self, centroid):
        g_centroids = self.retrieve_global_centroids()
        min_dist = float('inf')
        nearest_g_centroid = None  # initialize as []?
        for g_centroid in g_centroids:
            if euclidean(g_centroid, centroid) < min_dist:
                min_dist = euclidean(g_centroid, centroid)
                nearest_g_centroid = g_centroid
        return nearest_g_centroid

    def match_local_with_global_centroids(self):
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

    def aggr_updates(self, updates_per_centroid):
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
        return np.asarray(aggr_centroids)

    # ToDo: implement FedAvg.
    def aggr_fed_avg(self):
        aggr_centroids = []
        updates_weights = []
        nr_records_list = [local_update_tuple[1] for local_update_tuple in self.centroids_idxs_records]
        total_nr_records = sum(nr_records_list)

        # Obtain scaling factors. To be used for computing the weighted average.
        for local_update_tuple in self.centroids_idxs_records:
            curr_centroids = local_update_tuple[0]
            curr_nr_records = local_update_tuple[1]
            curr_idx = local_update_tuple[-1]  # to be used for validation step and contribution update.
            scaling_fac = self._obtain_scaling_factor(curr_nr_records, total_nr_records, len(nr_records_list))
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
                aggr_centroids.append(weighted_avgs)  # cannot do .tolist() as with simple averaging (above).
            else:
                aggr_centroids.append(self.retrieve_global_centroids()[i])  # should then put the global centroid.

        self.updated_centroids = np.asarray(aggr_centroids)
        return np.asarray(aggr_centroids)

    # placeholder function, may not be necessary in the end.
    def _obtain_scaling_factor(self, datasize, totalsize, nr_data_owners):
        scaling_factor = datasize / (totalsize / nr_data_owners)
        return scaling_factor

    # This should be a function that is run by the leaders after obtaining aggr_centroids from all committee members.
    def compute_new_global_centroids(self, aggr_centroids):
        g_centroids = self.retrieve_global_centroids()
        assert len(g_centroids) == len(aggr_centroids), "Number of global centroids not equal to aggregated centroids."
        new_g_centroids = []
        for i in range(len(g_centroids)):
            new_g_centroids.append(.5 * g_centroids[i] + .5 * aggr_centroids[i])  # Simple update rule for now.
        # print(self.validate_update(np.asarray(new_g_centroids)))
        self.updated_centroids = np.asarray(new_g_centroids)
        return np.asarray(new_g_centroids)

    def send_centroids(self, leader):
        assert leader.return_role() == "leader", "Supplied device is not a leader."
        leader.new_centroids.append(self.updated_centroids)
        leader.seen_committee_idxs.add(self.return_idx())

    def send_aggr_and_feedback(self):
        # requires update_contribution to alter a class variable (for Device type committee member).
        # this class variable can be retrieved here to send both the updated centroids and feedback per device.
        pass

    # update_contribution can be called s.t for idx_update in self.centroid_idxs: self.update_contribution(idx_update)
    # ToDo: rewrite this using the new centroids_idxs_records, rather than centroids_idxs. Also retrieve
    #  the most recent contribution value from blockchain.
    def update_contribution(self, idx_update):
        # idx_update is a tuple being device_idx, local centroids
        device_idx = idx_update[0]
        local_centroids = idx_update[1]
        score = self.validate_update(local_centroids)
        for peer in self.peer_list:
            if peer.return_idx() == device_idx:
                pass
                # ToDo: change this to update the (local) contribution value for the data owner. To be sent to the
                #  leader alongside the aggregated update s.t. they can put it in their proposed block.
                # using performance of global model to compare with for now
                # if score < self.validate_update(self.retrieve_global_centroids()):
                #     peer.contribution -= 1 else: peer.contribution += 1

        # recall: C(local t) = beta * (L(local) - L(global)) + (1-beta) * C(local t-1)
        # where we consider L(.) to be the silhouette score. Positive C if it improves, negative C if it worsens.

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

    # approve block effectively produces a vote for said block.
    def approve_block(self, block):  # i.e. if verify_block then approve_block
        msg = None
        # if the new block's centroids perform as well as or better than the previous block's centroids, we approve.
        if self.validate_update(block.get_data()['centroids']) >= \
                self.validate_update(self.retrieve_global_centroids()):
            msg = [self.sign(block.get_hash()), self.return_idx(), self.return_rsa_pub_key(), dt.datetime.now()]
        return msg

    def accept_propagated_block(self, propagated_block, source_idx):
        if source_idx not in self.black_list and propagated_block.get_hash() != self.obtain_latest_block().get_hash():
            self.append_block(propagated_block)
        else:
            print(f"Either {source_idx} is in {self.return_idx()}'s blacklist or "
                  f"the block was already added to {self.return_idx()}'s chain.")

    def request_to_download(self, block_to_download):
        for peer in self.peer_list:
            if peer.is_online() and self.is_online():
                # check whether the block is valid and has not already been added to the chain for current peer.
                if peer.verify_block(block_to_download) and \
                        peer.obtain_latest_block().get_hash() != block_to_download.get_hash():
                    peer.append_block(block_to_download)
            else:
                print(f"Either {peer.return_idx()} or {self.return_idx()} is currently offline and thus the block"
                      f"could not be downloaded.")

    def send_vote(self, vote, device_idx):
        leader_found = False
        for peer in self.peer_list:
            if peer.return_idx() == device_idx and peer.return_role() == 'leader':
                peer.received_votes.append(vote)
                leader_found = True
        if not leader_found:
            print(f"{device_idx} is not a leader node.")

    def return_candidate_blocks(self):
        return self.candidate_blocks

    # Used to reset variables at the start of a communication round (round-specific variables) for committee members.
    def reset_vars_committee_member(self):
        self.associated_data_owners_set = set()
        self.performances_this_round = {}
        self.committee_local_performance = float('-inf')
        self.obtained_local_centroids = []
        self.obtained_updates_unordered = {}
        self.updated_centroids = []
        self.centroids_idxs_records = []
        self.received_propagated_block = None
        self.candidate_blocks = []
        self.candidate_blocks_unordered = {}

    ''' leader '''

    def compute_update(self):
        obtained_centroids = np.asarray(self.new_centroids)
        g_centroids = self.retrieve_global_centroids()
        print(obtained_centroids.shape)  # in 'simple case': 5 comm_members each having 3 centroids having 2 dims.
        # i.e. we want for each centroid (0, 1, ..., k) to average the results over all committee members.
        # indexed like obtained_centroids[c][k][dim].
        updated_g_centroids = []
        # compute aggregates.
        for i in range(len(obtained_centroids[0])):
            updated_g_centroids.append(obtained_centroids[:, i].mean(axis=0))

        # compute new global centroids (simple update step).
        new_g_centroids = []
        assert len(updated_g_centroids) == len(g_centroids)
        for i in range(len(updated_g_centroids)):
            new_g_centroids.append(.5 * g_centroids[i] + .5 * updated_g_centroids[i])

        deltas = []
        for i in range(len(g_centroids)):
            deltas.append(euclidean(g_centroids[i], new_g_centroids[i]))
        stop_per_centroid = [delta < self.stop_condition for delta in deltas]
        print(deltas, all(stop_per_centroid))

        if all(stop_per_centroid):
            self._broadcast_stop_request()

        return np.asarray(new_g_centroids)

    def propose_block(self, new_g_centroids):
        previous_hash = self.obtain_latest_block().get_hash()
        data = dict()
        data['centroids'] = new_g_centroids
        block = Block(index=self.blockchain.get_chain_length() + 1, data=data, previous_hash=previous_hash,
                      miner_pubkey=self.return_rsa_pub_key(), produced_by=self.return_idx())
        block.set_signature(self.sign(block))
        self.proposed_block = block
        return block

    # Add the proposed block from leader to the committee members' candidate blocks.
    def broadcast_block(self, proposed_block):
        online_committee_members = self.return_online_committee_members()
        for committee_member in online_committee_members:
            committee_member.candidate_blocks.append(proposed_block)

    def _broadcast_stop_request(self):
        print('Stopping condition met! Requesting peers to stop global learning process...')
        pass

    # Propagate the proposed block (received majority vote) to all committee members.
    def propagate_block(self, block_to_propagate):
        committee = self.return_online_committee_members()
        for member in committee:
            member.accept_propagated_block(block_to_propagate, self.return_idx())

    def return_online_committee_members(self):
        online_committee_members = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "committee":
                    online_committee_members.add(peer)
        return online_committee_members

    # Q: Should this stay here as part of the leader functions, considering it is in fact a getter function?
    def return_received_votes(self):
        return self.received_votes

    # For block finalization, to prove that the leader has counted votes they add them to the proposed block before
    # appending.
    def serialize_votes(self):
        assert self.proposed_block is not None, f"{self.return_idx()} has not proposed a block yet."
        obtained_signatures = []
        vote_timestamps = []
        for vote in self.received_votes:
            obtained_signatures.append(vote[0])  # ToDo: figure out whether to verify the signature, given key.
            vote_timestamps.append(vote[-1])
        serialized_votes = pickle.dumps(obtained_signatures)
        self.proposed_block.set_vote_serial(bytearray(serialized_votes))
        self.proposed_block.set_block_time(max(vote_timestamps))
        signature = self.sign(self.proposed_block)
        self.proposed_block.set_signature(signature)

    def request_final_block_verification(self):
        assert self.proposed_block is not None, f"{self.return_idx()} has not proposed a block yet."
        comm_members = self.return_online_committee_members()
        verify_results = []
        for comm_member in comm_members:
            verify_results.append(comm_member.verify_block(self.proposed_block))
        return all(verify_results)  # returns False if any verification has failed.

    # checks for each committee member whether they successfully executed comm. steps and
    # if so, assigns positive feedback. Negative otherwise.
    # ToDo: implement reputation update, checking whether an aggregate was obtained from each committee member.
    def update_reputation(self, committee):
        online_committee_members = self.return_online_committee_members()
        for committee_member in online_committee_members:
            if committee_member.return_idx() not in self.seen_committee_idxs:
                cur_reputation = self.obtain_latest_block().get_data()['reputation'][committee_member.return_idx()]
                new_reputation = (cur_reputation[0] + 1, cur_reputation[1])
                pass
                # increment unsuccessful operations by one
            else:
                pass
                # increment successful operations by one

    # Used to reset variables at the start of a communication round (round-specific variables) for leaders.
    def reset_vars_leader(self):
        self.seen_committee_idxs = set()
        self.new_centroids = []
        self.proposed_block = None
        self.received_votes = []
        self.obtained_aggregated_unordered = {}
        self.obtained_aggregates_arrival_order_queue = {}


# Class to define and build each Device as specified by the parameters supplied. Returns a list of Devices.
class DevicesInNetwork(object):
    def __init__(self, is_iid, num_devices, num_malicious, network_stability, committee_wait_time, committee_threshold,
                 equal_link_speed, data_transmission_speed, equal_computation_power, check_signature, stop_condition,
                 bc=None, dataset=None):
        self.dataset = dataset
        self.is_iid = is_iid
        self.num_devices = num_devices
        self.num_malicious = num_malicious
        self.network_stability = network_stability
        self.equal_link_speed = equal_link_speed
        self.equal_computation_power = equal_computation_power
        self.data_transmission_speed = data_transmission_speed
        # committee
        self.committee_wait_time = committee_wait_time
        self.committee_threshold = committee_threshold
        self.check_signature = check_signature
        self.stop_condition = stop_condition
        # blockchain
        if bc is not None:
            self.blockchain = copy.copy(bc)
        else:
            self.blockchain = Blockchain()
        # divide data
        self.devices_set = {}
        self._dataset_allocation()

    # For now, let us allocate a (simple) testing dataset.
    # ToDo: change this to use more sensible data, but still be dependent on said data (i.e. values)
    def _dataset_allocation(self):
        # read dataset
        train_data, labels = data_utils.load_data(num_devices=self.num_devices, is_iid=self.is_iid, samples=500)

        # then create individual devices, each having their local dataset.
        for i in range(self.num_devices):
            # divide the data equally
            local_dataset = [train_data[i], labels[i]]

            device_idx = f'device_{i + 1}'
            a_device = Device(device_idx, local_dataset, self.network_stability, self.committee_wait_time,
                              self.committee_threshold, self.equal_link_speed, self.data_transmission_speed,
                              self.equal_computation_power, self.check_signature, self.stop_condition,
                              bc=self.blockchain)
            self.devices_set[device_idx] = a_device
            print(f"Creation of device having idx {device_idx} is done.")
