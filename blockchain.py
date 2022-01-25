import hashlib
import time
import copy

from block import Block


# ToDo: rewrite mine, Block.mine and allow the genesis block to be created within main.py
class Blockchain:
    def __init__(self, difficulty=0):
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

    def replace_chain(self, blocks):
        self.blocks = copy.copy(blocks)

    ''' getters '''

    def get_most_recent_block(self):  # genesis block is created at init, so we can always return self.blocks[-1].
        if len(self.blocks) > 0:
            return self.blocks[-1]
        else:
            print("There are no blocks on this chain.")
            return None

    def get_chain_length(self):
        return len(self.blocks)

    def get_chain_structure(self):
        return self.blocks
