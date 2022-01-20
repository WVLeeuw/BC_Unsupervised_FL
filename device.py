import numpy as np
import torch
import time
from hashlib import sha256
from Crypto.PublicKey import RSA

from blockchain import Blockchain


class Device:
    def __init__(self):
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.role = None

    ''' setters '''

    def generate_rsa_key(self):
        kp = RSA.generate(bits=1024)
        self.modulus = kp.n
        self.private_key = kp.d
        self.public_key = kp.e

    def assign_role(self):
        # non-equal prob, depends on proportion worker:committee:leaders.
        pass

    # following assignments are for hard assignments
    def assign_worker_role(self):
        self.role = "worker"

    def assign_committee_role(self):
        self.role = "committee"

    def assign_leader_role(self):
        self.role = "leader"

    ''' getters '''

    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    ''' functions '''

    def sign(self, msg):
        h = int.from_bytes(sha256(msg).digest(), byteorder='big')
        signature = pow(h, self.private_key, self.public_key)
        return signature

    def verify_signature(self, msg):  # Requires using other devices' public key.
        pass

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
        pass

    ''' data owner '''

    def kmeans(self):
        pass

    def local_update(self):
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


# Class to define and build each Device as specified by the parameters supplied. Returns a list of Devices.
class DevicesInNetwork(object):
    def __init__(self, is_iid):
        self.is_iid = is_iid

    def dataset_balanced_allocation(self):
        pass
