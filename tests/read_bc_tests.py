import json

from blockchain import Blockchain
import block

import copy

import os

bc_folders = '../blockchains/'


def test_individual_block_read():
    date = '03242022_173023'
    with open(f"{bc_folders + date}/round_63_block_12.json") as file:
        cur_block = json.load(file)
    # block = Block(**cur_block)  # This would work if all keys in jsonfile were in block params.
    cur_block = block.fromJSON(cur_block)
    print(type(cur_block), cur_block)


def test_multiple_to_bc():
    date = '03242022_173023'
    blocks = []
    for f in os.listdir(bc_folders + date):
        if f.endswith('.json'):
            cur_block = json.load(open(bc_folders + date + '/' + f))
            blocks.append(block.fromJSON(cur_block))

    blocks = sorted(blocks, key=lambda x: x.index, reverse=False)
    chain = Blockchain()

    for b in blocks:
        chain.append_block(b)

    print(chain.is_chain_valid(), chain.get_chain_length())
