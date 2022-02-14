import numpy as np
import time
import random
import copy

from hashlib import sha256
from Crypto.PublicKey import RSA
from sklearn import cluster
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean

from blockchain import Blockchain
import KMeans
from utils import data_utils


class Device:
    def __init__(self, idx, ds, network_stability, committee_wait_time, committee_threshold, equal_link_speed,
                 base_data_transmission_speed, equal_computation_power, check_signature, bc=None, model=None):
        # Identifier
        self.idx = idx

        # Trust values
        self.reputation = (1, 1)
        self.contribution = 0

        # Datasets
        self.dataset = ds[0]
        self.labels = ds[1]

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
        self.generate_rsa_key()
        # Data owners
        self.performance_this_round = float('-inf')
        self.local_update_time = None
        # Committee members
        self.associated_data_owners_set = set()
        self.performances_this_round = {}
        # self.associated_leaders = set()
        self.local_centroids = []
        self.committee_wait_time = committee_wait_time
        self.committee_threshold = committee_threshold
        self.committee_local_performance = None
        self.received_propagated_block = None
        # Leaders
        # self.leader_associated_members = set()
        self.mined_block = None

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

    def assign_leader_role(self):
        self.role = "leader"

    # if anything goes wrong during the learning process of a certain device, can reset
    def initialize_kmeans_model(self, n_clusters=3, verbose=False):
        self.model = cluster.KMeans(n_clusters=n_clusters, max_iter=10)

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

    def return_reputation(self):
        return self.reputation

    def return_contribution(self):
        return self.contribution

    # Returns the performance (using some performance measure) of a data owner for the current round
    def return_performance(self):
        return self.performance_this_round

    # Returns the performances recorded from local updates by a committee member
    def return_performances(self):
        return self.performances_this_round

    ''' functions '''

    ''' not role-specific '''

    def sign(self, msg):
        h = int.from_bytes(sha256(msg).digest(), byteorder='big')
        signature = pow(h, self.private_key, self.public_key)
        return signature

    def verify_signature(self, msg):
        if self.check_signature:
            # ToDo: ensure the following are included in msg.
            modulus = msg['rsa_pub_key']["modulus"]
            pub_key = msg['rsa_pub_key']["pub_key"]
            signature = msg['signature']
            h = int.from_bytes(sha256(msg).digest(), byteorder='big')
            h_signed = pow(signature, pub_key, modulus)
            if h == h_signed:
                print("The signature is valid and the message has been verified.")
                return True
            else:
                print("The signature is invalid and the message was not recorded.")
                return False
        print("The message has been verified.")  # placeholder
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
        return latest_block.data['centroids']

    def add_block(self, block_to_add):
        self.blockchain.mine(block_to_add)

    def obtain_latest_block(self):
        return self.blockchain.get_most_recent_block()

    ''' data owner '''

    def local_update(self):
        # Retrieve the newest block and specifically the centroids recorded in it.
        g_centroids = self.retrieve_global_centroids()
        # Build a new (local) model using the centroids.
        self.model = cluster.KMeans(n_clusters=g_centroids.shape[0], init=g_centroids, n_init=1, max_iter=10)
        self.model.fit(self.dataset)  # and perform the update.

    def retrieve_local_centroids(self):
        local_centroids = self.model.cluster_centers_
        # send update to associated committee member
        return local_centroids

    # Used to reset variables at the start of a communication round (round-specific variables) for data owners
    def reset_vars_data_owner(self):
        self.performance_this_round = float('-inf')

    ''' committee member '''

    # ToDo: rewrite s.t. we also take the device idx for update_contribution.
    def validate_update(self, local_centroids):
        self.model = cluster.KMeans(n_clusters=local_centroids.shape[0], init=local_centroids, n_init=1, max_iter=10)
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
            for centroids in self.local_centroids:
                for centroid in centroids:
                    if np.array_equal(global_centroid, self.find_nearest_global_centroid(centroid)):
                        to_aggregate.append(centroid)
                # print("Found average silhouette score of " + str(self.validate_update(centroids)))
            print("Found " + str(len(to_aggregate)) + " local centroids with which to update the global centroid.")
            updates_per_centroid.append(to_aggregate)
        return updates_per_centroid

    def aggr_updates(self, updates_per_centroid):
        aggr_centroids = []
        for i in range(len(updates_per_centroid)):
            if not isinstance(updates_per_centroid[i], np.ndarray):  # convert to np.ndarray to use numpy functions.
                updates_per_centroid[i] = np.array(updates_per_centroid[i])
            if len(updates_per_centroid[i]) > 0:  # we can use np.mean function to take the averages.
                avgs = updates_per_centroid[i].mean(axis=0)  # taking simple average of 'columns' for now.
                aggr_centroids.append(avgs.tolist())
            else:  # committee member received no updates for this centroid
                aggr_centroids.append(self.retrieve_global_centroids()[i])  # should put the global centroid here.
        return np.asarray(aggr_centroids)

    def compute_new_global_centroids(self, aggr_centroids):
        g_centroids = self.retrieve_global_centroids()
        assert len(g_centroids) == len(aggr_centroids), "Number of global centroids not equal to aggregated centroids."
        new_g_centroids = []
        for i in range(len(g_centroids)):
            new_g_centroids.append(.5 * g_centroids[i] + .5 * aggr_centroids[i])  # Simple update rule for now.
        # print(self.validate_update(np.asarray(new_g_centroids)))
        return np.asarray(new_g_centroids)

    def send_aggr_and_feedback(self):
        pass

    def return_online_data_owners(self):
        online_data_owners = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "data owner":
                    online_data_owners.add(peer)
        return online_data_owners

    def update_contribution(self, idx_update):
        # idx_update is a tuple being device_idx, local centroids
        device_idx = idx_update[0]
        local_centroids = idx_update[1]
        score = self.validate_update(local_centroids)
        for peer in self.peer_list:
            if peer.return_idx() == device_idx:
                # using performance of global model to compare with for now
                if score < self.validate_update(self.retrieve_global_centroids()):
                    peer.contribution -= 1
                else:
                    peer.contribution += 1

    # Used to reset variables at the start of a communication round (round-specific variables) for committee members.
    def reset_vars_committee_member(self):
        self.associated_data_owners_set = set()
        self.performances_this_round = {}
        self.committee_local_performance = float('-inf')
        self.received_propagated_block = None

    def verify_block(self):
        pass

    def approve_block(self):  # i.e. if verify_block then approve_block
        pass

    ''' leader '''

    def compute_update(self):
        pass

    def update_feedback(self):
        pass

    def propose_block(self):
        pass

    def broadcast_block(self):  # may be unnecessary, considering obtain_latest_block also exists.
        pass

    def return_online_committee_members(self):
        online_committee_members = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "committee member":
                    online_committee_members.add(peer)
        return online_committee_members

    def update_reputation(self):
        pass

    # Used to reset variables at the start of a communication round (round-specific variables) for leaders.
    def reset_vars_leader(self):
        self.mined_block = None


# Class to define and build each Device as specified by the parameters supplied. Returns a list of Devices.
class DevicesInNetwork(object):
    def __init__(self, is_iid, num_devices, num_malicious, network_stability, committee_wait_time, committee_threshold,
                 equal_link_speed, data_transmission_speed, equal_computation_power, check_signature, bc=None,
                 dataset=None):
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
        # blockchain
        if bc is not None:
            self.blockchain = copy.copy(bc)
        else:
            self.blockchain = Blockchain()
        # divide data
        self.devices_set = {}
        self._dataset_allocation()

    # For now, let us allocate a (simple) testing dataset.
    def _dataset_allocation(self):
        # read dataset
        train_data, labels = data_utils.create_blobs()

        # then divide across devices
        # train_data = dataset[0]
        # test_data = dataset[1]

        # data_size_train = len(train_data) // self.num_devices
        # data_size_test = test_data // self.num_devices

        # Depending on is_iid, we allocate an equal amount of records to each device.
        dfs = np.array_split(train_data, self.num_devices)
        dfs_labels = np.array_split(labels, self.num_devices)
        if not self.is_iid:
            for i in range(self.num_devices):
                # divide the data equally
                local_dataset = [dfs[i], dfs_labels[i]]

                device_idx = f'device_{i + 1}'
                a_device = Device(device_idx, local_dataset, self.network_stability, self.committee_wait_time,
                                  self.committee_threshold, self.equal_link_speed, self.data_transmission_speed,
                                  self.equal_computation_power, self.check_signature, bc=self.blockchain)
                self.devices_set[device_idx] = a_device
                print(f"Creation of device having idx {device_idx} is done.")
        else:
            raise NotImplementedError  # ToDo: simulate non-iid distribution of data.
