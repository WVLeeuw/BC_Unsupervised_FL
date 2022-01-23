import hashlib
import datetime as dt
from Crypto.PublicKey import RSA


# We start with a PoW block type.
class Block:  # Can put block as a dictionary. Though data should always be reserved for (model) parameters.
    def __init__(self, index, timestamp, data, previous_hash, nonce=0, signature=None):
        self.hash = hashlib.sha256()
        self.index = index
        self.previous_hash = previous_hash  # refer to the last block (link them together)
        self.nonce = nonce  # only applicable in PoW, otherwise this is just the block counter.
        self.data = data  # all updates and feedback should be in block
        self.timestamp = timestamp  # does this go well? At block creation, should cal dt.datetime.now()]
        self.signature = signature

    def mine(self, difficulty):
        self.hash.update(str(self).encode('utf-8'))
        while int(self.hash.hexdigest(), 16) > 2 ** (256 - difficulty):  # while hash larger than difficulty required
            self.nonce += 1
            self.hash = hashlib.sha256()
            self.hash.update(str(self).encode('utf-8'))

    def __str__(self):
        return "{} {} {}".format(self.previous_hash.hexdigest(), self.data, self.nonce)

    def nonce_increment(self):
        self.nonce += 1

    ''' signature functions '''

    def remove_signature(self):
        self.signature = None

    def add_signature(self, signing_key):
        pass

    ''' checks '''

    def is_valid(self):
        pass

    ''' getters '''

    def get_index(self):
        return self.index

    def get_timestamp(self):
        return self.timestamp

    def get_hash(self):
        return self.hash

    def get_previous_hash(self):
        return self.previous_hash

    def get_data(self):
        return self.data

    def get_signature(self):
        return self.signature

    ''' setters '''

    def add_verified_transaction(self, transaction):
        self.data.append(transaction)
