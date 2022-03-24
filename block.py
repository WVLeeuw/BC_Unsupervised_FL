import hashlib
import datetime as dt
import json

from Crypto.PublicKey import RSA


class Block:  # Can put block as a dictionary. Though data should always be reserved for (model) parameters.
    def __init__(self, data, previous_hash, index=None, nonce=0, signature=None, produced_by=None, miner_pubkey=None):
        self.hash = hashlib.sha256()
        self.index = index
        self.previous_hash = previous_hash  # refer to the last block (link them together)
        self.nonce = nonce  # for proof-of-work, if difficulty > 0.
        self.data = data  # all parameters (and feedback) should be in data
        self.timestamp = str(dt.datetime.now())
        self.signature = signature
        # leader specific
        self.produced_by = produced_by
        self.miner_pubkey = miner_pubkey
        self.vote_serialization = []
        self.block_time = self.timestamp

    def mine(self, difficulty):
        self.hash.update(str(self).encode('utf-8'))
        while int(self.hash.hexdigest(), 16) > 2 ** (256 - difficulty):  # while hash larger than difficulty required
            self.nonce += 1
            self.hash = hashlib.sha256()
            self.hash.update(str(self).encode('utf-8'))

    # N.B. this string representation is also used for hashing/signing.
    def __str__(self):
        # First ensure that each value can actually be put as is.
        prev_hash = self.previous_hash
        if not isinstance(self.previous_hash, str):
            prev_hash = self.previous_hash.hexdigest()
        return f"index: {self.index}, \n" \
               f"previous hash: {prev_hash}, \n" \
               f"data: {self.data}, \n" \
               f"produced by: {self.produced_by}, \n" \
               f"votes serial: {self.vote_serialization}. \n" \
               f"block generation time: {self.timestamp}, \n" \
               f"block finalization time: {self.block_time}"

    def toJSON(self):
        self.data['centroids'] = self.data['centroids'].tolist()
        self.hash = self.hash.hexdigest()
        self.previous_hash = self.previous_hash.hexdigest()
        self.vote_serialization = list(self.vote_serialization)
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=2)

    # ToDo: read block (or multiple blocks) from file as .json.
    def fromJSON(self):
        pass

    ''' signature functions '''

    def remove_signature(self):
        self.signature = None

    def set_signature(self, signature):
        self.signature = signature

    def set_vote_serial(self, vote_serialization):
        self.vote_serialization = vote_serialization

    def set_block_time(self, block_time):
        self.block_time = str(block_time)

    ''' getters '''

    def get_hash(self):
        return self.hash

    def get_index(self):
        return self.index

    def get_previous_hash(self):
        return self.previous_hash

    def get_data(self):
        return self.data

    def get_timestamp(self):
        return self.timestamp

    def get_signature(self):
        return self.signature

    def get_produced_by(self):
        return self.produced_by

    def get_miner_pk(self):
        return self.miner_pubkey

    def get_vote_serial(self):
        return self.vote_serialization
