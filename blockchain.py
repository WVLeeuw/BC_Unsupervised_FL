import hashlib
import time
import copy

from block import Block


# ToDo: rewrite mine, Block.mine and allow the genesis block to be created within main.py
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

    def _add_to_chain(self, block):
        if self.proof_of_work(block):
            self.blocks.append(block)
        else:
            print("The block could not be appended.")

    # N.B. block should already be defined, meaning it should include a reference to the previous block.
    def mine(self, block):
        block.mine(self.difficulty)
        self._add_to_chain(block)

    def create_genesis_block(self):  # dummy block for now, later we can add some sensible parameters in data
        h = hashlib.sha256()
        h.update(''.encode('utf-8'))
        genesis = Block(data='Genesis', previous_hash=h)
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
