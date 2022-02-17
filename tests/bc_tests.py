from blockchain import Blockchain
from block import Block

from hashlib import sha256


def test_bc_genesis():
    bc = Blockchain()
    bc.create_genesis_block()
    print(str(bc.get_chain_structure()[0]) == str(bc.get_most_recent_block()))

    # Sanity check.
    # print(sha256(str(bc.get_most_recent_block()).encode('utf-8')).hexdigest() ==
    #       bc.get_most_recent_block().hash.hexdigest())
    #
    # # Another sanity check.
    # h = sha256()
    # h.update(str(bc.get_most_recent_block()).encode('utf-8'))
    # print(h.hexdigest() == bc.get_most_recent_block().hash.hexdigest())


def test_block_propagation():
    bc = Blockchain()
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()
    # Test whether we acquire the same hash if we rehash the most recent block.
    print(sha256(str(bc.get_most_recent_block()).encode('utf-8')).hexdigest() == prev_block.get_hash().hexdigest())

    # # Additional sanity check.
    # new_hash = sha256()
    # new_hash.update(str(bc.get_most_recent_block()).encode('utf-8'))
    # hexdig = new_hash.hexdigest()
    # print(hexdig == prev_block.hash.hexdigest())
    #
    # # More sanity checks.
    # print(prev_block.get_hash().hexdigest() == prev_block.hash.hexdigest())
    # print(sha256(str(bc.get_most_recent_block()).encode('utf-8')).hexdigest())
    # print(prev_block.hash.hexdigest())
    # print(bc.get_chain_length())

    # Test whether we can add a new block using that hash.
    block = Block(data='test_block', previous_hash=prev_block.get_hash())
    bc.mine(block)
    added_properly = bc.get_chain_length()
    if added_properly:
        print(str(bc.get_most_recent_block()))


def test_nonce_increment():
    bc = Blockchain(difficulty=0)
    bc.create_genesis_block()
    prev_block = bc.get_most_recent_block()

    block1 = Block(data={}, previous_hash=prev_block.get_hash())
    bc.mine(block1)

    recent_block = bc.get_most_recent_block()
    block2 = Block(data={}, previous_hash=recent_block.get_hash())
    print(sha256(str(block2).encode('utf-8')).hexdigest())
    bc.mine(block2)

    print(str(bc.get_chain_structure()[-2]), str(bc.get_chain_structure()[-1]))
