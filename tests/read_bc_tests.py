import json

from blockchain import Blockchain
from block import Block

import copy

import os

bc_folders = '../blockchains/'


def test_bc_read():
    date = '03242022_173023'
    with open(f"{bc_folders+date}/round_63_block_12.json") as file:
        block = json.load(file)
    print(type(block))
    print(block)
