import hashlib
import time
import copy
import numpy as np

from block import Block


class Blockchain:
    def __init__(self, difficulty=0):
        self.difficulty = difficulty  # N.B. We set the difficulty at the chain-level.
        self.blocks = []

    def proof_of_work(self, block):
        h = hashlib.sha256()
        h.update(str(block).encode('utf-8'))
        return (block.hash.hexdigest() == h.hexdigest()) and \
               (int(h.hexdigest(), 16) < 2 ** (256 - self.difficulty)) and \
               (block.previous_hash == self.blocks[-1].hash)

    def add_to_chain(self, block):
        if self.proof_of_work(block):
            self.blocks.append(block)
        else:
            print("The block could not be appended.")

    def append_block(self, block):
        self.blocks.append(copy.copy(block))

    def mine(self, block):
        block.mine(self.difficulty)
        self.add_to_chain(block)

    def create_genesis_block(self, data=None):
        h = hashlib.sha256()
        h.update(''.encode('utf-8'))
        # data = np.zeros((2, 2))
        genesis = Block(index=1, data=data, previous_hash=h, produced_by="Genesis")
        genesis.mine(self.difficulty)
        self.blocks.append(genesis)

    def replace_chain(self, blocks):
        self.blocks = copy.copy(blocks)

    # Checks whether the chain is valid, should suffice in terms of 0-difficulty proof-of-work.
    def is_chain_valid(self):
        block_index = 1
        previous_block = self.blocks[0]

        while block_index < len(self.blocks):
            block = self.blocks[block_index]
            # Check if previous hash of the current block is the same as the hash of the previous block
            if block.previous_hash != previous_block.hash:
                return False

            previous_block = block
            block_index += 1

        return True

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
