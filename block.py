import hashlib
import datetime as dt
from Crypto.PublicKey import RSA


class Block:  # Can put block as a dictionary. Though data should always be reserved for (model) parameters.
    def __init__(self, data, previous_hash, index=None, nonce=0, signature=None, mined_by=None, miner_pubkey=None):
        self.hash = hashlib.sha256()
        self.index = index
        self.previous_hash = previous_hash  # refer to the last block (link them together)
        self.nonce = nonce  # for proof-of-work
        self.data = data  # all parameters (and feedback) should be in data
        self.timestamp = str(dt.datetime.now())
        self.signature = signature
        # leader specific
        self.mined_by = mined_by
        self.miner_pubkey = miner_pubkey

    def mine(self, difficulty):
        self.hash.update(str(self).encode('utf-8'))
        while int(self.hash.hexdigest(), 16) > 2 ** (256 - difficulty):  # while hash larger than difficulty required
            self.nonce += 1
            self.hash = hashlib.sha256()
            self.hash.update(str(self).encode('utf-8'))

    # ToDo: improve way of showing the block contents.
    def __str__(self):
        return "{} {} {} {}".format(self.previous_hash.hexdigest(), self.data, self.nonce, self.timestamp)

    # def nonce_increment(self):
    #     self.nonce += 1

    ''' signature functions '''

    def remove_signature(self):
        self.signature = None

    def set_signature(self, signature):
        self.signature = signature

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

    def get_mined_by(self):
        return self.mined_by

    def get_miner_pk(self):
        return self.miner_pubkey
