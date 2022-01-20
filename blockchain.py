import hashlib
import time

from block import Block


class Blockchain:
    def __init__(self, difficulty):
        self.difficulty = difficulty  # N.B. We set the difficulty at the chain-level for now.
        self.blocks = []
        self.pool = []  # pool of transactions to complete
        self.create_genesis_block()

    def proof_of_work(self, block):
        h = hashlib.sha256()
        h.update(str(block).encode('utf-8'))
        return (block.hash.hexdigest() == h.hexdigest()) and \
               (int(h.hexdigest(), 16) < 2 ** (256 - self.difficulty)) and \
               (block.previous_hash == self.blocks[-1].hash)

    def add_to_chain(self, block):
        if self.proof_of_work(block):
            self.blocks.append(block)

    def add_to_pool(self, data):
        self.pool.append(data)

    def mine(self):
        if len(self.pool) > 0:
            data = self.pool.pop()
            block = Block(data, self.blocks[-1].hash)
            block.mine(self.difficulty)
            self.add_to_chain(block)

    def create_genesis_block(self):  # dummy block for now, later we can add some sensible parameters in data
        h = hashlib.sha256()
        h.update(''.encode('utf-8'))
        genesis = Block('Genesis', h)
        genesis.mine(self.difficulty)
        self.blocks.append(genesis)

    ''' validity functions '''

    ''' forking functions '''

    ''' getters '''

    def get_most_recent_block(self):  # genesis block is created at init, so we can always return self.blocks[-1].
        return self.blocks[-1]

    def get_chain_length(self):
        return len(self.blocks)

    def get_chain_structure(self):
        return self.blocks
