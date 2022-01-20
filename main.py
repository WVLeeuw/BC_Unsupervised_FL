from block import Block
from blockchain import Blockchain
from device import Device, DevicesInNetwork

import os
import sys
import argparse
import random
import time
from datetime import datetime

date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_folder_path = f"logs/{date_time}"
model_snapshots_folder = "models"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='BCFL_kmeans_Simulation')

# debug attributes
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')

# FL attributes
# parser.add_argument('-B', '--batchsize', type=int, default=10, help='batch size used in local training')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communication rounds')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='number of devices in the simulation')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help='number of malicious devices')
parser.add_argument('-iid', '--IID', type=int, default=0, help='whether to allocate data in iid setting')
parser.add_argument('-gc', '--num_global_centroids', type=int, default=5, help='number of centroids in the globally '
                                                                               'trained model')
parser.add_argument('-lc', '--num_local_centroids', type=int, default=3, help='number of centroids in locally trained '
                                                                              'models')
parser.add_argument('-le', '--num_local_epochs', type=int, default=1, help='number of epochs to perform for the '
                                                                           'acquisition of local update')

# BC attributes (consensus & committee parameters)
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6, help='amount of rounds after which to kick a '
                                                                           'malicious data owner')
parser.add_argument('-lo', '--lazy_knock_out_rounds', type=int, default=10, help='amount of rounds after which to '
                                                                                 'kick a non-responsive data owner')
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0, help='threshold value of a difference in '
                                                                                  'accuracy with which to identify '
                                                                                  'malicious data owners')

# distributed system attributes

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help='number of devices assigned to the roles '
                                                                            'of data owner, committee member and '
                                                                            'committee leader respectively')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='whether to check signatures, used to save '
                                                                          'time or to assume trust')


if __name__ == '__main__':

    # get arguments
    args = parser.parse_args()
    args = args.__dict__

    # set up from scratch
    # 0. create directory for log files
    if not os.path.isdir(log_folder_path):
        os.mkdir(log_folder_path)

    # 1. save arguments and parameters used
    with open(f'{log_folder_path}/args_used.txt', 'w') as f:
        f.write("Command line arguments used -\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nAll arguments used -\n")
        for arg_name, arg in args.items():
            f.write(f'\n--{arg_name} {arg}')

    # 2. create folder to save models in, if it does not already exist
    if not os.path.isdir(model_snapshots_folder):
        os.mkdir(model_snapshots_folder)
    os.mkdir(f'{model_snapshots_folder}/{date_time}')

    # 3. get number of roles required in the network
    roles_requirement = args['hard_assign'].split(',')

    # 4. check arguments eligibility
    num_devices = args['num_devices']
    num_malicious = args['num_malicious']

    if num_malicious:
        if num_malicious > num_devices:
            sys.exit("ERROR: The number of malicious devices cannot exceed the total number of devices.")
        else:
            print(f'Malicious nodes and total devices set to {num_malicious}:{num_devices}')

    # 5. create devices in the network

    # 6. register devices and initialize global parameters

    # 7. build log files, to be filled during execution

    # BCFL-KMeans starts here
    # ...
