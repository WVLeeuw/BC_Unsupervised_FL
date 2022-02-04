from blockchain import Blockchain
from block import Block

from hashlib import sha256


def test_bc_genesis():
    bc = Blockchain()
    bc.create_genesis_block()
    print(str(bc.get_chain_structure()[0]))


def test_block_propagation():
    bc = Blockchain()
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()
    # Test whether we acquire the same hash if we rehash the most recent block.
    print(sha256(str(bc.get_most_recent_block()).encode('utf-8')).hexdigest() == prev_block.get_hash().hexdigest())

    # Test whether we can add a new block using that hash.
    block = Block(data='test_block', previous_hash=prev_block.get_hash())
    bc.mine(block)
    added_properly = bc.get_chain_length()
    if added_properly:
        print(str(bc.get_most_recent_block()))


def test_nonce_increment():
    bc = Blockchain(difficulty=5)
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()

    block1 = Block(data='first_test_block', previous_hash=prev_block.get_hash())
    bc.mine(block1)

    recent_block = bc.get_most_recent_block()
    block2 = Block(data='second_test_block', previous_hash=recent_block.get_hash())
    bc.mine(block2)

    print(str(bc.get_chain_structure()[-2]), str(bc.get_chain_structure()[-1]))
