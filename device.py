import numpy as np
import torch
import time
import random
from hashlib import sha256
from Crypto.PublicKey import RSA

import KMeans
from blockchain import Blockchain
from KMeans import KMeans


class Device:
    def __init__(self, idx, train_ds, test_ds, network_stability, committee_wait_time, committee_threshold,
                 pow_difficulty, equal_link_speed, base_data_transmission_speed, equal_computation_power,
                 check_signature, knock_out_rounds, lazy_knock_out_rounds, model=None):
        # Identifier
        self.idx = idx

        # Datasets
        self.train_ds = train_ds
        self.test_ds = test_ds

        # P2P network variables
        self.online = True
        self.peer_list = set()
        self.network_stability = network_stability
        self.equal_link_speed = equal_link_speed
        self.base_data_transmission_speed = base_data_transmission_speed
        self.equal_computation_power = equal_computation_power
        self.knock_out_rounds = knock_out_rounds
        self.lazy_knock_out_rounds = lazy_knock_out_rounds

        # BC variables
        self.pow_difficulty = pow_difficulty
        self.check_signature = check_signature
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
        self.associated_leaders = set()
        self.committee_wait_time = committee_wait_time
        self.committee_threshold = committee_threshold
        self.committee_local_performance = None
        self.received_propagated_block = None
        # Leaders
        self.leader_associated_members = set()
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

    # Function assigning the roles to devices with some probability and constraints. May be done elsewhere.
    # def assign_role(self):
    #     # non-equal prob, depends on proportion worker:committee:leaders.
    #     pass

    # following assignments are for hard assignments
    def assign_data_role(self):
        self.role = "data owner"
        self.model = KMeans.init_kmeans(n_clusters=3, verbose=True)

    def assign_committee_role(self):
        self.role = "committee"

    def assign_leader_role(self):
        self.role = "leader"

    # if anything goes wrong during the learning process of a certain device, can reset
    def initialize_kmeans_model(self, n_clusters, verbose):
        self.model = KMeans.init_kmeans(n_clusters=n_clusters, verbose=verbose)

    ''' getters '''

    def return_idx(self):
        return self.idx

    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def return_blockchain_obj(self):
        return self.blockchain

    def return_pk(self):
        return self.public_key

    def is_online(self):
        return self.online

    ''' functions '''

    def sign(self, msg):
        h = int.from_bytes(sha256(msg).digest(), byteorder='big')
        signature = pow(h, self.private_key, self.public_key)
        return signature

    # Requires using other devices' public key.
    def verify_signature(self, msg, pk):
        pass

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        # else:
        #     self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        # else:
        #     self.peer_list.difference_update(peers_to_remove)

    def switch_online_status(self):
        cur_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.online = True
            # if the device has been offline, we update their peer list and resync the chain
            if not cur_status:
                print(f"{self.idx} has come back online.")
                # update peer list
                # resync chain
        else:
            self.online = False
            print(f"{self.idx} has gone offline.")
        return self.online

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

    def check_pow_proof(self, block_to_check):
        pass

    def check_chain_validity(self, chain_to_check):
        pass

    def resync_chain(self, mining_consensus):  # describes how to sync after failed validity or fork
        pass

    def add_block(self, block_to_add):
        pass

    ''' not role-specific '''

    def obtain_latest_block(self):
        return self.blockchain.get_most_recent_block()

    ''' data owner '''

    def kmeans(self):
        pass

    def local_update(self, model):
        pass

    def send_update(self):
        pass

    ''' committee member '''

    def validate_update(self):
        pass

    def aggr_update(self):
        pass

    def send_aggr(self):
        pass

    def send_feedback(self):
        pass

    def return_online_associated_devices(self):
        online_associated_devices = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "data owner":
                    online_associated_devices.add(peer)
        return online_associated_devices

    # send_packet can be the combination of send_aggr and send_feedback

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

    def append_block(self):
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


# Class to define and build each Device as specified by the parameters supplied. Returns a list of Devices.
class DevicesInNetwork(object):
    def __init__(self, is_iid):
        self.is_iid = is_iid

    def dataset_balanced_allocation(self):
        pass
